import os
import time

import numpy as np
import pandas as pd
import torch

from cifar10.async_grads.shard_scheduler import ShardAssignment
from cifar10.load_data import preload_cifar10_to_ram
from cifar10.model import cifar10_get_model
from ddp import DDPClient
from ddp.pickle_utils import log, send_msg


class CIFAR10Worker(DDPClient):
    """
    Worker async que devuelve delta de pesos.
    """

    def __init__(self, host, port):
        super().__init__(host, port)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.dataset = None
        self.test_dataset = None

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

    def get_shard(self, test=False):
        N = len(self.test_dataset) if test else len(self.dataset)

        self.last_epoch = self.assignment.epoch
        rng = np.random.default_rng(seed=self.assignment.epoch)
        indices = rng.permutation(N)

        start = self.assignment.start
        end = min(start + self.assignment.length, N)

        if test:
            start = rng.integers(0, N)
            end = min(start + self.assignment.length, N)

        shard = indices[start : min(end, N)]

        return shard

    def get_batch(self, test=False):
        if self.assignment is None:
            return None, None

        shard = self.get_shard(test)

        n = (len(shard) // self.assignment.batch_size) * self.assignment.batch_size
        shard = shard[:n]

        for i in range(0, n, self.assignment.batch_size):
            indixes = shard[i : i + self.assignment.batch_size]

            X, y = self.test_dataset[indixes] if test else self.dataset[indixes]
            X, y = X.to(self.device), y.to(self.device)
            yield X, y

    def test(self):
        self.model.eval()
        eval_loss, eval_correct, eval_total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in self.get_batch(test=True):
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

        for X, y in self.get_batch():
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
            gray = payload["gray"]
            normalize = payload["normalize"]
            conv = payload["conv"]
            lr = payload["lr"]

            self.model, self.criterion, self.optimizer = cifar10_get_model(
                gray=gray, conv=conv, lr=lr, device=self.device
            )

            self.dataset = preload_cifar10_to_ram(
                train=True,
                gray=gray,
                normalize=normalize,
            )

            self.test_dataset = preload_cifar10_to_ram(
                train=False,
                gray=gray,
                normalize=normalize,
            )

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
