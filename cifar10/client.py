import os
import time

import numpy as np
import pandas as pd
import torch
from torchinfo import summary

from ddp import DDPClient
from ddp.pickle_utils import send_msg

from .load_data import preload_cifar10_to_ram
from .model import Cifar10Model


class CIFAR10Worker(DDPClient):
    """
    Cliente worker para el entrenamiento distribuido de CIFAR-10.
    """

    def __init__(self, host, port, gray=True, normalize=True, conv=False, lr=0.01):
        super().__init__(host, port)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Cifar10Model(gray=gray, conv=conv).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        summary(self.model, input_size=(1, 1 if gray else 3, 32, 32))

        self.dataset = preload_cifar10_to_ram(
            train=True,
            gray=gray,
            normalize=normalize,
        )

        self.rank = 0
        self.world_size = 1

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

    def get_batch(self, epoch):
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

    def _register_handlers(self):
        """
        Registra los manejadores de mensajes del servidor.
        Ejecuta las funciones correspondientes cuando se reciben mensajes del servidor.
        """

        @self.on("assign")
        def on_assign(msg):
            payload = msg["payload"]

            self.rank = payload["rank"]
            self.world_size = payload["world_size"]

        @self.on("weights")
        def on_weights(msg):
            state = msg["payload"]

            state_dict = self.model.state_dict()

            for k in state_dict:
                state_dict[k] = torch.tensor(state[k])

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

            batch_idx = self.get_batch(epoch)
            X, y = self.dataset[batch_idx]

            X = X.to(self.device)
            y = y.to(self.device)

            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model(X)
            loss = self.criterion(logits, y)

            loss.backward()

            grads = {
                name: param.grad.detach().cpu().numpy().astype(np.float32)
                for name, param in self.model.named_parameters()
                if param.grad is not None
            }

            acc = float((logits.argmax(1) == y).float().mean().item())
            loss = float(loss.item())

            elapse = time.perf_counter() - t0
            throughput = len(X) / elapse

            print(
                f"Worker {self.rank}: epoch={epoch}, acc={acc:.4f}, loss={loss:.4f}, elapse={elapse:.4f}, throughput={throughput:.4f}"
            )

            self.metrics.loc[len(self.metrics)] = [acc, loss, elapse, throughput]

            send_msg(
                self._sock,
                {
                    "type": "result",
                    "payload": {
                        "grads": grads,
                        "loss": loss,
                        "accuracy": acc,
                    },
                },
            )
