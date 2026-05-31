import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ddp import DDPClient
from ddp.pickle_utils import log, send_msg

from .shard import AsyncShardSampler, IndexedDataset
from .shard_scheduler import ShardAssignment


class AsyncGradWorker(DDPClient):
    """
    Worker ASGD que recibe pesos, calcula gradientes sobre un shard y devuelve
    gradientes al servidor. El worker no actualiza pesos localmente.
    """

    def __init__(self, host, port, save_path: str | None = None):
        super().__init__(host, port)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.dataset = None
        self.test_dataset = None

        self.batch_size = 128
        self.assignment: ShardAssignment | None = None
        self.stop = False

        self.sampler = AsyncShardSampler()
        self.indexed_dataset = None
        self.loader = None

        self.test_indexed_dataset = None
        self.test_loader = None

        self.metrics = pd.DataFrame(
            columns=["loss", "accuracy", "elapse", "throughput"]
        )

        self._register_handlers()

    def save_metrics(self):
        if self.save_path is None:
            return

        os.makedirs(self.save_path, exist_ok=True)

        self.metrics.to_excel(
            os.path.join(self.save_path, f"metrics_{self._worker_id}.xlsx"),
            index=False,
        )
        description = self.metrics.describe(percentiles=[0.1, 0.5, 0.9])
        description.to_excel(
            os.path.join(self.save_path, f"metrics_{self._worker_id}_desc.xlsx"),
            index=True,
        )

    def load_samplers(self, preload: bool = False):
        """
        Crea loaders una sola vez. En cada step solo se cambian indices.

        preload=True fuerza num_workers=0, util cuando dataset ya esta en RAM.
        """
        self.indexed_dataset = IndexedDataset(self.dataset)
        self.loader = DataLoader(
            self.indexed_dataset,
            batch_size=self.batch_size,
            num_workers=0 if preload else 2,
            persistent_workers=False,
            prefetch_factor=None if preload else 2,
            pin_memory=torch.cuda.is_available(),
        )

        self.test_indexed_dataset = IndexedDataset(self.test_dataset)
        self.test_loader = DataLoader(
            self.test_indexed_dataset,
            batch_size=self.batch_size,
            num_workers=0 if preload else 2,
            persistent_workers=False,
            prefetch_factor=None if preload else 2,
            pin_memory=torch.cuda.is_available(),
        )

    def _ensure_loaders(self):
        if self.loader is None or self.test_loader is None:
            self.load_samplers(preload=False)

    def test(self):
        self._ensure_loaders()
        self.model.eval()
        t0 = time.perf_counter()
        test_loss, test_correct, test_total = 0.0, 0, 0

        indices = np.arange(len(self.test_dataset))
        self.test_indexed_dataset.set_indices(indices)

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)

                outputs = self.model(X)
                loss = self.criterion(outputs, y)

                test_loss += loss.item() * y.size(0)
                test_correct += (outputs.argmax(1) == y).sum().item()
                test_total += y.size(0)

                print(
                    f"Eval loss: {loss.item():.4f} "
                    f"| correct: {test_correct}/{test_total}",
                    end="\r",
                )

        if test_total == 0:
            return 0.0, 0.0, 0.0

        elapsed = time.perf_counter() - t0
        return test_loss / test_total, test_correct / test_total, elapsed

    def train(self, t0, w_global=None):
        self._ensure_loaders()
        self.model.train()

        total_loss = torch.tensor(0.0)
        total_correct = torch.tensor(0.0)
        total_samples = torch.tensor(0.0)
        n_batches = 0

        indices = self.sampler.get_indices(self.dataset, self.assignment)
        self.indexed_dataset.set_indices(indices)

        self.optimizer.zero_grad()

        for X, y in self.loader:
            X, y = X.to(self.device), y.to(self.device)

            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()

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
                    f" size: {y.size(0)}",
                    end="\r",
                )

        if self.scheduler is not None:
            self.scheduler.step()

        grads = {
            name: (param.grad / n_batches).detach().cpu().numpy().astype(np.float32)
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

        if total_samples.item() == 0:
            return grads, 0.0, 0.0, 0.0, 0

        avg_acc = total_correct / total_samples
        avg_loss = total_loss / total_samples
        elapse = time.perf_counter() - t0
        throughput = total_samples / elapse if elapse > 0 else torch.tensor(0.0)

        return (
            grads,
            avg_loss.item(),
            avg_acc.item(),
            elapse,
            throughput.item(),
            int(total_samples.item()),
        )

    def _register_handlers(self):
        @self.on("stop")
        def on_stop(msg):
            log.info(f"Recibido mensaje de stop: {msg}")
            self.stop = True

        @self.on("config")
        def on_config(msg):
            pass

        @self.on("metrics")
        def on_metrics(msg):
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

        @self.on("step")
        def on_step(msg):
            if self.stop:
                return

            payload = msg.get("payload", None)
            if payload is None:
                log.warning("No hay payload en step")
                return

            epoch = payload.get("epoch", 0)
            k_iter = payload.get("iter", 0)
            state = payload.get("weights", None)
            assignment = payload.get("assignment", None)

            if state is None or assignment is None:
                log.warning("No hay weights o assignment en step")
                return

            state_dict = {k: torch.tensor(v) for k, v in state.items()}
            self.model.load_state_dict(state_dict)
            self.assignment = assignment
            self.batch_size = assignment.batch_size

            t0 = time.perf_counter()
            grads, loss, accuracy, elapse, throughput, samples = self.train(t0)

            log.info(
                f"Worker: {self._worker_id} | epoch={epoch} "
                f"| acc={accuracy:.4f} | loss={loss:.4f} "
                f"| elapsed={elapse:.4f} | throughput={throughput:.4f}"
            )

            self.metrics.loc[len(self.metrics)] = [
                loss,
                accuracy,
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
                        "grads": grads,
                        "samples": samples,
                        "loss": loss,
                        "accuracy": accuracy,
                        "iter_sent": k_iter,
                        "shard_idx": self.assignment.shard_idx,
                    },
                },
            )

        @self.on("test")
        def on_test(msg):
            if self.stop:
                return

            payload = msg.get("payload", None)
            if payload is None:
                log.warning("No hay payload en test")
                return

            k_iter = payload.get("iter", 0)
            state = payload.get("weights", None)

            if state is None:
                log.warning("No hay weights en test")
                return

            state_dict = {k: torch.tensor(v) for k, v in state.items()}
            self.model.load_state_dict(state_dict)

            loss, accuracy, elapsed = self.test()

            log.info(
                f"Test: iter={k_iter} | loss={loss:.4f} "
                f"| accuracy={accuracy:.4f} | elapsed={elapsed:.4f}"
            )

            send_msg(
                self._sock,
                {
                    "type": "test_result",
                    "worker_id": self._worker_id,
                    "payload": {
                        "iter": k_iter,
                        "loss": loss,
                        "accuracy": accuracy,
                        "elapsed": elapsed,
                    },
                },
            )
