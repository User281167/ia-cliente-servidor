import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ddp import DDPClient
from ddp.pickle_utils import log, send_msg
from utils.format_time import format_elapse

from .load_data import ShardSampler, TinyImageNetLazy
from .model import get_tiny_imagenet_model


class TinyImangeNetWorker(DDPClient):
    """
    Cliente worker para el entrenamiento distribuido de Tiny ImageNet.
    """

    def __init__(self, host, port):
        super().__init__(host, port)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[torch.nn.Module] = None
        self.criterion: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None

        self.dataset: Optional[TinyImageNetLazy] = None
        self.eval_dataset: Optional[TinyImageNetLazy] = None
        self.sampler: Optional[ShardSampler] = None
        self.eval_sampler: Optional[ShardSampler] = None
        self.ready = False

        self.rank: Optional[int] = None
        self.world_size: Optional[int] = None
        self.batch_size: Optional[int] = None

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

    def _register_handlers(self):
        """
        Registra los manejadores de mensajes del servidor.
        Ejecuta las funciones correspondientes cuando se reciben mensajes del servidor.
        """

        @self.on("config")
        def on_config(msg):
            log.info(f"Recibido mensaje de configuración: {msg}")

            payload = msg["payload"]
            lr = payload["lr"]
            epochs = payload["epochs"]
            self.batch_size = payload["batch_size"]

            self.model, self.criterion, self.optimizer, self.scheduler = (
                get_tiny_imagenet_model(lr=lr, epochs=epochs, device=self.device)
            )

            self.dataset = TinyImageNetLazy()
            self.eval_dataset = TinyImageNetLazy(split="valid")
            self.ready = True

        @self.on("assign")
        def on_assign(msg):
            payload = msg["payload"]
            new_rank = payload["rank"]
            new_world_size = payload["world_size"]

            # Solo recrear sampler si realmente cambió algo
            if new_rank != self.rank or new_world_size != self.world_size:
                self.rank = new_rank
                self.world_size = new_world_size

                self.sampler = ShardSampler(
                    dataset_size=len(self.dataset),
                    rank=self.rank,
                    world_size=self.world_size,
                    batch_size=self.batch_size,
                )

                self.eval_sampler = ShardSampler(
                    dataset_size=len(self.eval_dataset),
                    rank=self.rank,
                    world_size=self.world_size,
                    batch_size=self.batch_size,
                )

        @self.on("weights")
        def on_weights(msg):
            if not self.ready:
                return

            state = msg["payload"]
            state_dict = {k: torch.tensor(v) for k, v in state.items()}
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
            Entrenmiento con minibatch, realizar step en cada batch y enviar delta de pesos
            """
            if not self.ready:
                log.warning("Ignorando paso: modelo no asignado")
                return

            epoch = msg["epoch"]
            w_global = {k: v.clone() for k, v in self.model.state_dict().items()}

            # test dataset
            self.model.eval()
            eval_loss, eval_correct, eval_total = 0.0, 0, 0
            eval_loader = self.eval_sampler.get_loader(epoch, self.eval_dataset)

            with torch.no_grad():
                for X, y in eval_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)
                    eval_loss += loss.item() * y.size(0)
                    eval_correct += (outputs.argmax(1) == y).sum().item()
                    eval_total += y.size(0)

                    print(
                        f"Eval loss: {loss.item():.4f}, correct: {eval_correct}/{eval_total}",
                        end="\r",
                    )

            # Train
            t0 = time.perf_counter()
            self.model.train()
            loader = self.sampler.get_loader(epoch, self.dataset)

            total_loss, total_correct, total_samples = 0.0, 0, 0
            n_batches = 0

            for X, y in loader:
                t_init = time.perf_counter()

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
                n_batches += 1

                elapsed = time.perf_counter() - t_init

                if n_batches % 10 == 0:
                    print(
                        f"Batch {n_batches}, loss: {loss.item():.4f}, acc: {total_correct / total_samples:.4f}, elapsed: {format_elapse(elapsed)}",
                        end="\r",
                    )

            if self.scheduler is not None:
                self.scheduler.step()

            # Δw = w_local - w_global
            w_local = self.model.state_dict()
            delta = {
                k: (w_local[k] - w_global[k]).cpu().numpy().astype(np.float16)
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
                        # eval distribuido
                        "eval_loss": eval_loss,  # suma, no promedio
                        "eval_correct": eval_correct,
                        "eval_total": eval_total,
                    },
                },
            )
