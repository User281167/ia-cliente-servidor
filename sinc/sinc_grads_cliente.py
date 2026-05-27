import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ddp import DDPClient
from ddp.pickle_utils import log, send_msg

from .shard_sampler import DistributedEpochSampler


class SincGradWorker(DDPClient):
    """
    Cliente worker para el entrenamiento distribuido con sincronización de gradientes.
    """

    def __init__(self, host, port, save_path):
        super().__init__(host, port)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.dataset = None
        self.test_dataset = None

        self.rank = 0
        self.world_size = 1
        self.batch_size = 128

        self.sampler = None
        self.loader = None

        self.test_sampler = None
        self.test_loader = None

        self.metrics = pd.DataFrame(
            columns=["loss", "accuracy", "elapse", "throughput"]
        )

        self._register_handlers()

    def save_metrics(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        self.metrics.to_excel(
            os.path.join(self.save_path, f"metrics_{self.rank}.xlsx"), index=False
        )
        description = self.metrics.describe(percentiles=[0.1, 0.5, 0.9])
        description.to_excel(
            os.path.join(self.save_path, f"metrics_{self.rank}_desc.xlsx"), index=True
        )

    def test(self, seed):
        # test dataset
        self.model.eval()
        eval_loss, eval_correct, eval_total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)

                outputs = self.model(X)
                loss = self.criterion(outputs, y)

                eval_loss += loss.item() * y.size(0)
                eval_correct += (outputs.argmax(1) == y).sum().item()
                eval_total += y.size(0)

                print(
                    f"Eval loss: {loss.item():.4f} "
                    f"| correct: {eval_correct}/{eval_total}",
                    end="\r",
                )

        return eval_loss, eval_correct, eval_total

    def train(self, seed, t0, w_global=None):
        # train
        self.model.train()
        total_loss = torch.tensor(0.0)
        total_correct = torch.tensor(0.0)
        total_samples = torch.tensor(0.0)
        n_batches = 0

        self.optimizer.zero_grad()

        for X, y in self.loader:
            X, y = X.to(self.device), y.to(self.device)

            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()  # acumula gradientes crudos, sin dividir

            total_loss += loss.item() * y.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            n_batches += 1

            if n_batches % 10 == 0:
                print(
                    f"Batch {n_batches}| "
                    f"loss: {loss.item():.4f} | "
                    f"acc: {total_correct / total_samples:.4f}",
                    end="\r",
                )

        if self.scheduler is not None:
            self.scheduler.step()

        grads = {
            name: param.grad.detach().cpu().numpy().astype(np.float32)
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

        avg_acc = total_correct / total_samples
        avg_loss = total_loss / total_samples

        elapse = time.perf_counter() - t0
        throughput = total_samples / elapse

        return (
            grads,
            avg_loss.item(),
            avg_acc.item(),
            elapse,
            throughput,
            total_samples.item(),
        )

    def load_samplers(self, preload=False):
        """Carga los samplers y los dataloaders para el entrenamiento y prueba."""
        self.sampler = DistributedEpochSampler(len(self.dataset), self.batch_size)

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=2 if not preload else 0,
            persistent_workers=not preload,
            prefetch_factor=2 if not preload else None,
            pin_memory=torch.cuda.is_available(),
        )

        self.test_sampler = DistributedEpochSampler(
            len(self.test_dataset), self.batch_size
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=2 if not preload else 0,
            persistent_workers=not preload,
            prefetch_factor=2 if not preload else None,
            pin_memory=torch.cuda.is_available(),
        )

    def _register_handlers(self):
        """
        Registra los manejadores de mensajes del servidor.
        Ejecuta las funciones correspondientes cuando se reciben mensajes del servidor.
        """

        @self.on("config")
        def on_config(msg):
            pass

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

        @self.on("metrics")
        def on_metrics(msg):
            """
            Enviar métricas al servidor
            """
            send_msg(
                self._sock,
                {
                    "type": "metrics",
                    "payload": {
                        "data_frame": self.metrics,
                        "rank": self.rank,
                    },
                },
            )

        @self.on("step")
        def on_step(msg):
            """
            Manejador para el mensaje "step".
            Recibe un lote de datos y realiza una iteración de entrenamiento.
            No realiza optimización ni actualización de pesos.
            """
            t0 = time.perf_counter()

            epoch = msg["epoch"]
            seed = msg.get("seed", epoch)

            self.test_sampler.set_epoch(seed, self.rank, self.world_size)
            self.sampler.set_epoch(seed, self.rank, self.world_size)

            eval_loss, eval_correct, eval_total = self.test(seed)
            grads, avg_loss, avg_acc, elapse, throughput, total_samples = self.train(
                seed, t0
            )

            log.info(
                f"Worker {self.rank}: epoch={epoch} | "
                f"acc={avg_acc:.4f} | loss={avg_loss:.4f} | "
                f"elapse={elapse:.4f} | throughput={throughput:.4f}"
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
                        "grads": grads,
                        "samples": total_samples,
                        "loss": avg_loss,
                        "accuracy": avg_acc,
                        # test
                        "eval_loss": eval_loss,  # suma, no promedio
                        "eval_correct": eval_correct,
                        "eval_total": eval_total,
                    },
                },
            )
