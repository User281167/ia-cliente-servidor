import os
import time

import numpy as np
import pandas as pd
import torch

from ddp import DDPClient
from ddp.pickle_utils import log, send_msg
from ddp.shard_scheduler import ShardAssignment
from tiny_imagenet.load_data import AsyncShardSampler, TinyImageNetLazy
from tiny_imagenet.resnet18.model import get_tiny_imagenet_model


class Worker(DDPClient):
    """
    Worker async que devuelve delta de pesos.
    """

    def __init__(self, host, port):
        super().__init__(host, port)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.dataset = None
        self.test_dataset = None
        self.shard_sampler = AsyncShardSampler()

        self.last_epoch = 0
        self.assignment: ShardAssignment | None = None

        self.stop = False
        self.metrics = pd.DataFrame(
            columns=["loss", "accuracy", "elapse", "throughput"]
        )

        self._register_handlers()

    def save_metrics(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        self.metrics.to_excel(
            os.path.join(path, f"metrics_{np.random.randint(0, 1000000)}.xlsx"),
            index=False,
        )
        description = self.metrics.describe(percentiles=[0.1, 0.5, 0.9])
        description.to_excel(
            os.path.join(path, f"metrics_desc_{np.random.randint(0, 1000000)}.xlsx"),
            index=True,
        )

    def test(self):
        self.model.eval()
        eval_loss, eval_correct, eval_total = 0.0, 0, 0

        loader = self.shard_sampler.get_loader(
            self.test_dataset, self.assignment, test=True
        )

        with torch.no_grad():
            for X, y in loader:
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

        if eval_total == 0:
            eval_total = 1

        return eval_loss / eval_total, eval_correct / eval_total

    def train(self, t0):
        self.model.train()
        w_global = {k: v.clone() for k, v in self.model.state_dict().items()}
        total_loss = torch.tensor(0.0)
        total_correct = torch.tensor(0.0)
        total_samples = torch.tensor(0.0)
        n_batches = 0

        loader = self.shard_sampler.get_loader(self.dataset, self.assignment)

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            n_batches += 1

            if n_batches % 10 == 0:
                print(
                    f"Batch {n_batches}, loss: {loss.item():.4f}, acc: {total_correct / total_samples:.4f}, total: {total_samples:.0f}",
                    end="\r",
                )

        self.scheduler.step()

        w_local = self.model.state_dict()
        delta = {
            k: (w_local[k] - w_global[k]).cpu().numpy().astype(np.float32)
            for k in w_global
        }

        avg_acc = total_correct / total_samples
        avg_loss = total_loss / total_samples

        elapse = time.perf_counter() - t0
        throughput = total_samples / elapse

        return (
            delta,
            avg_acc,
            avg_loss,
            elapse,
            throughput.item(),
            int(total_samples.item()),
        )

    def _register_handlers(self):
        @self.on("stop")
        def on_stop(msg):
            log.info(f"Recibido mensaje de stop: {msg}")
            self.stop = True

        @self.on("metrics")
        def on_metrics(msg):
            log.info(f"Recibido mensaje de metricas: {msg}")

            send_msg(
                self._sock,
                {
                    "type": "metrics",
                    "worker_id": self._worker_id,
                    "payload": {
                        "data_frame": self.metrics,
                    },
                },
            )

        @self.on("config")
        def on_config(msg):
            log.info(f"Recibido mensaje de configuracion: {msg}")

            payload = msg["payload"]
            lr = payload["lr"]

            self.model, self.criterion, self.optimizer, self.scheduler = (
                get_tiny_imagenet_model(lr=lr, device=self.device)
            )

            self.dataset = TinyImageNetLazy(split="train")
            self.test_dataset = TinyImageNetLazy(split="valid")

            self.stop = False
            send_msg(self._sock, {"type": "ready", "worker_id": self._worker_id})

        @self.on("step")
        def on_step(msg):
            if self.stop:
                return

            log.info("Recibido paso")
            payload = msg.get("payload", None)

            if payload is None:
                log.warning("No hay payload en el paso, ignorando")
                return

            epoch = payload.get("epoch", 0)
            k_iter = payload.get("iter", 0)
            state = payload.get("weights", None)
            assignment = payload.get("assignment", None)

            if state is None or assignment is None:
                log.warning("No hay estado o asignacion en el paso, ignorando")
                return

            state_dict = {k: torch.tensor(v) for k, v in state.items()}
            self.model.load_state_dict(state_dict)
            self.assignment = assignment

            t0 = time.perf_counter()

            eval_loss, eval_correct = self.test()
            delta, acc, loss, elapse, throughput, samples = self.train(t0)

            log.info(
                f"Worker: epoch={epoch}, "
                f"acc={acc:.4f}, test_acc={eval_correct:.4f}, "
                f"loss={loss:.4f}, test_loss={eval_loss:.4f}, "
                f"elapse={elapse:.4f}, throughput={throughput:.4f}"
            )

            self.metrics.loc[len(self.metrics)] = [
                eval_loss,
                eval_correct,
                elapse,
                throughput,
            ]

            if self.stop:
                return

            send_msg(
                self._sock,
                {
                    "type": "result",
                    "worker_id": self._worker_id,
                    "payload": {
                        "delta": delta,
                        "acc": acc.item(),
                        "loss": loss.item(),
                        "iter_sent": k_iter,
                        "shard_idx": self.assignment.shard_idx,
                        "samples": samples,
                        "test_acc": eval_correct,
                        "test_loss": eval_loss,
                    },
                },
            )
