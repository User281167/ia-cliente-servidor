import os
import time

import numpy as np
import pandas as pd
import torch

from ddp import DDPClient
from ddp.pickle_utils import log, send_msg

from .load_data import preload_cifar10_to_ram
from .model import cifar10_get_model


class CIFAR10Worker(DDPClient):
    """
    Cliente worker para el entrenamiento distribuido de CIFAR-10.
    """

    def __init__(self, host, port):
        super().__init__(host, port)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.dataset = None

        self.rank = 0
        self.world_size = 1
        self.batch_size = 128

        self.metrics = pd.DataFrame(
            columns=["loss", "accuracy", "elapse", "throughput"]
        )

        self._register_handlers()

    def save_metrics(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        self.metrics.to_excel(
            os.path.join(path, f"metrics_{self.rank}.xlsx"), index=False
        )
        description = self.metrics.describe(percentiles=[0.1, 0.5, 0.9])
        description.to_excel(
            os.path.join(path, f"metrics_{self.rank}_desc.xlsx"), index=True
        )

    def get_shard(self, epoch):
        """
        Obtiene un lote de datos para la época dada.
        Realizar shuffle global y shard (toma de datos) local.
        Evita que los datos sean siempre los mismos en cada época y que se solapen entre workers.
        """
        N = len(self.dataset)

        rng = np.random.default_rng(seed=epoch)

        # shuffle global
        # shard del worker
        indices = rng.permutation(N)
        shard = indices[self.rank :: self.world_size]

        return shard

    def get_batch(self, epoch):
        """
        Minibatches para no entrenar con todo el conjunto y explotar la memoria.
        """
        shard = self.get_shard(epoch)
        # evitar el batch incompleto
        n = (len(shard) // self.batch_size) * self.batch_size
        shard = shard[:n]

        for i in range(0, n, self.batch_size):
            yield shard[i : i + self.batch_size]

    def _register_handlers(self):
        """
        Registra los manejadores de mensajes del servidor.
        Ejecuta las funciones correspondientes cuando se reciben mensajes del servidor.
        """

        @self.on("config")
        def on_config(msg):
            log.info(f"Recibido mensaje de configuración: {msg}")

            payload = msg["payload"]
            gray = payload["gray"]
            normalize = payload["normalize"]
            conv = payload["conv"]
            lr = payload["lr"]
            self.batch_size = payload["batch_size"]

            self.model, self.criterion, self.optimizer = cifar10_get_model(
                gray=gray, conv=conv, lr=lr, device=self.device
            )

            self.dataset = preload_cifar10_to_ram(
                train=True,
                gray=gray,
                normalize=normalize,
            )

        @self.on("assign")
        def on_assign(msg):
            payload = msg["payload"]

            self.rank = payload["rank"]
            self.world_size = payload["world_size"]

        @self.on("weights")
        def on_weights(msg):
            state = msg["payload"]
            state_dict = {k: torch.tensor(v) for k, v in state.items()}
            self.model.load_state_dict(state_dict)

        @self.on("step")
        def on_step(msg):
            """
            Manejador para el mensaje "step".
            Recibe un lote de datos y realiza una iteración de entrenamiento.
            No realiza optimización ni actualización de pesos.
            """
            t0 = time.perf_counter()

            epoch = msg["epoch"]
            w_global = {k: v.clone() for k, v in self.model.state_dict().items()}

            total_loss, total_correct, total_samples = 0.0, 0, 0
            steps_done = 0
            n_batches = 0

            for batch_idx in self.get_batch(epoch):
                X, y = self.dataset[batch_idx]
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X)

                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                _, preds = torch.max(outputs, 1)
                total_loss += loss.item() * y.size(0)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
                steps_done += 1
                n_batches += 1

                if n_batches % 10 == 0:
                    print(
                        f"Batch {n_batches}, loss: {loss.item():.4f}, acc: {total_correct / total_samples:.4f}",
                        end="\r",
                    )

            # Δw = w_local - w_global
            w_local = self.model.state_dict()
            delta = {
                k: (w_local[k] - w_global[k]).cpu().numpy().astype(np.float32)
                for k in w_global
            }

            avg_acc = total_correct / total_samples
            avg_loss = total_loss / total_samples

            elapse = time.perf_counter() - t0
            throughput = total_samples / elapse

            log.info(
                f"Worker {self.rank}: epoch={epoch}, acc={avg_acc:.4f}, loss={avg_loss:.4f}, elapse={elapse:.4f}, throughput={throughput:.4f}"
            )

            self.metrics.loc[len(self.metrics)] = [
                avg_loss,
                avg_acc,
                elapse,
                throughput,
            ]

            send_msg(
                self._sock,
                {
                    "type": "result",
                    "payload": {
                        "delta": delta,
                        "samples": total_samples,
                        "loss": avg_loss,
                        "accuracy": avg_acc,
                    },
                },
            )
