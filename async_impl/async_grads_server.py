import os
import threading
import time

import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import MulticlassConfusionMatrix

from ddp import DDPAsyncServer
from ddp.logger import log
from ddp.message import DDPMessage
from ddp.pickle_utils import send_msg
from utils import plot_grid, time_wrapper

from .shard_scheduler import ShardScheduler


class AsyncGradServer(DDPAsyncServer):
    """
    Servidor ASGD generico.

    Algoritmo:
      worker recibe x^(k-delta), shard
      worker devuelve gradiente
      server aplica x^(k+1) = x^k - gamma_k * grad
      gamma_k = lr / (1 + staleness)
      server envia pesos frescos + siguiente shard al mismo worker
    """

    def __init__(
        self,
        data_len: int,
        epochs: int = 20,
        lr: float = 0.001,
        shard_size: int = 5000,
        batch_size: int = 128,
        max_staleness: int = 10,
        test_each: int = 10,
        min_workers: int = 1,
        config: dict | None = None,
        save_path: str | None = None,
    ):
        if config is None:
            config = {
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
            }

        config.setdefault("top5", False)
        self.compute_top5 = config["top5"]

        super().__init__(min_workers=min_workers, worker_config=config)

        self.lr = lr
        self.batch_size = batch_size
        self.shard_size = shard_size
        self.max_staleness = max_staleness
        self.test_each = test_each
        self.epochs = epochs
        self.save_path = save_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.criterion = self.optimizer = self.scheduler = None
        self.test_loader = None

        self._scheduler = ShardScheduler(data_len, shard_size, batch_size)
        self.k = 0
        self._k_lock = threading.Lock()
        self._last_test_worker_idx = -1

        self.metrics = pd.DataFrame(
            columns=[
                "loss",
                "accuracy",
                "top5_accuracy",
                "grad_norm",
                "staleness",
                "gamma",
                "elapsed",
            ]
        )
        self.test_metrics = pd.DataFrame(
            columns=[
                "iter",
                "worker_id",
                "loss",
                "accuracy",
                "top5_accuracy",
                "elapsed",
            ]
        )

        self._register_event_handlers()

    def _remove_dead(self, wids: list[int]) -> None:
        for wid in wids:
            self._scheduler.requeue(wid)

        super()._remove_dead(wids)

    def _gamma(self, staleness: int) -> float:
        return self.lr / (1.0 + staleness)

    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * y.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        return total_loss / total, correct / total

    def evaluate_classification(self, num_classes: int = 10):
        self.model.eval()
        correct, total = 0, 0
        confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(
            self.device
        )

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                confusion_matrix.update(predicted, labels)

        return correct / total, confusion_matrix.compute().cpu()

    def _get_state_numpy(self) -> dict:
        return {
            k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in self.model.state_dict().items()
        }

    def _apply_gradient(self, grads: dict, gamma: float) -> float:
        """
        x^{k+1} = x^k − γ_k · ∇f(x^{k-δ}; ξ)

        gamma ya incorpora la corrección por staleness: γ / (1 + δ).

        return:
            Norma L2 del gradiente
        """
        gnorm = 0.0

        for name, param in self.model.named_parameters():
            if name not in grads:
                continue

            g = torch.as_tensor(
                grads[name],
                dtype=param.dtype,
                device=param.device,
            )
            param.grad = g.detach()
            gnorm += torch.linalg.vector_norm(g.float()).item() ** 2

        old_lrs = [group["lr"] for group in self.optimizer.param_groups]

        for group in self.optimizer.param_groups:
            group["lr"] = gamma

        self.optimizer.step()
        self.optimizer.zero_grad()

        for group, old_lr in zip(self.optimizer.param_groups, old_lrs):
            group["lr"] = old_lr

        return gnorm**0.5

    def _send_step_to(self, wid: int, state: dict, k: int) -> None:
        with self._workers_lock:
            sock = self._workers.get(wid)

        if sock is None:
            log.error(f"Worker {wid} no conectado, skip step")
            return

        assignment = self._scheduler.next_shard(wid)

        try:
            send_msg(
                sock,
                DDPMessage.msg(
                    "step",
                    iter=k,
                    epoch=assignment.epoch,
                    weights=state,
                    assignment=assignment,
                ),
            )
        except Exception as e:
            log.error(f"No se pudo enviar step a worker {wid}: {e}")
            self._remove_dead([wid])

    def _send_test(
        self,
        wid: int,
        state: dict,
        k: int,
    ) -> None:
        with self._workers_lock:
            sock = self._workers.get(wid)

        if not sock:
            return

        try:
            send_msg(
                sock,
                DDPMessage.msg(
                    "test",
                    iter=k,
                    weights=state,
                ),
            )
        except Exception as e:
            log.error(f"No se pudo enviar test a worker {wid}: {e}")
            self._remove_dead([wid])

    def _register_event_handlers(self) -> None:
        @self.on("ready")
        def _handle_ready(msg: dict) -> None:
            wid = msg["worker_id"]

            with self._k_lock:
                state = self._get_state_numpy()
                k = self.k

            self._send_step_to(wid, state, k)

        @self.on("result")
        def _handle_result(msg: dict) -> None:
            wid = msg["worker_id"]
            payload = msg["payload"]
            t0 = time.perf_counter()

            grads = payload["grads"]
            samples = payload.get("samples", 0)
            loss = payload.get("loss", float("nan"))
            accuracy = payload.get("accuracy", float("nan"))
            top5_accuracy = payload.get("top5_accuracy", float("nan"))
            iter_sent = payload.get("iter_sent", self.k)
            shard_idx = payload.get("shard_idx", None)

            with self._k_lock:
                k_now = self.k
                staleness = k_now - iter_sent
                gamma = self._gamma(staleness)

                if staleness <= self.max_staleness:
                    grad_norm = self._apply_gradient(grads, gamma)
                else:
                    grad_norm = float("nan")

                self.k += 1
                fresh_state = self._get_state_numpy()
                k_new = self.k

                send_test = k_new % self.test_each == 0

                if staleness <= self.max_staleness:
                    self.metrics.loc[len(self.metrics)] = [
                        loss,
                        accuracy,
                        top5_accuracy,
                        grad_norm,
                        staleness,
                        gamma,
                        time.perf_counter() - t0,
                    ]

            if send_test:
                self._send_test(wid, fresh_state, k_new)
            else:
                self._send_step_to(wid, fresh_state, k_new)

            if shard_idx is not None:
                self._scheduler.complete(wid, shard_idx)

            if k_now % 10 == 0 and staleness <= self.max_staleness:
                txt = (
                    f"[k={k_now}] epoch={self._scheduler.current_epoch}/{self.epochs}"
                    f" | worker={wid} | staleness={staleness} | gamma={gamma:.6f} "
                    f" | samples={samples} | loss={loss:.4f} | accuracy={accuracy:.4f} "
                    f" | grad_norm={grad_norm:.4f}"
                )

                if self.compute_top5:
                    txt += f" | top5={top5_accuracy:.4f} "

                log.info(txt)

        @self.on("test_result")
        def _handle_test_result(msg: dict) -> None:
            wid = msg["worker_id"]
            payload = msg["payload"]

            self.test_metrics.loc[len(self.test_metrics)] = [
                payload.get("iter", self.k),
                wid,
                payload.get("loss", float("nan")),
                payload.get("accuracy", float("nan")),
                payload.get("top5_accuracy", float("nan")),
                payload.get("elapsed", float("nan")),
            ]

            txt = (
                f"[test k={payload.get('iter', self.k)}]: | worker={wid} | "
                f"loss={payload.get('loss', float('nan')):.4f} | "
                f"accuracy={payload.get('accuracy', float('nan')):.4f} | "
                f"elapsed={payload.get('elapsed', float('nan')):.4f}"
            )

            if self.compute_top5:
                txt += f" | top5 acc={payload.get('top5_accuracy', float('nan')):.4f}"

            log.info(txt)

            with self._k_lock:
                k_new = self.k + 1
                self.k = k_new
                fresh_state = self._get_state_numpy()

            self._send_step_to(wid, fresh_state, k_new)

        @self.on("metrics")
        def _handle_metrics(msg: dict) -> None:
            if self.save_path is None:
                return

            wid = msg["worker_id"]
            payload = msg["payload"]
            df = pd.DataFrame(payload["data_frame"])

            df.to_excel(os.path.join(self.save_path, f"metrics_{wid}.xlsx"))
            df.describe(percentiles=[0.1, 0.5, 0.9]).to_excel(
                os.path.join(self.save_path, f"description_{wid}.xlsx"),
                index=True,
            )

    @time_wrapper
    def train(self) -> None:
        while self._scheduler.current_epoch < self.epochs:
            time.sleep(1)

        log.info(f"Entrenamiento completado - k={self.k} iteraciones totales")

    def results(self) -> None:
        save_path = self.save_path

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.metrics.to_excel(os.path.join(save_path, "metrics_server.xlsx"))
            self.test_metrics.to_excel(
                os.path.join(save_path, "test_metrics_server.xlsx"),
                index=False,
            )
            self.metrics.describe(percentiles=[0.1, 0.5, 0.9]).to_excel(
                os.path.join(save_path, "description_server.xlsx"), index=True
            )

        n = len(self.metrics)
        train_labels = ["Loss", "Accuracy"]
        history = [
            [self.metrics["loss"][i], self.metrics["accuracy"][i]] for i in range(n)
        ]

        if self.compute_top5:
            train_labels.append("Top-5 Accuracy")

            for i in range(n):
                history[i].append(self.metrics["top5_accuracy"][i])

        train_labels.append("Grad Norm")

        for i in range(n):
            history[i].append(self.metrics["grad_norm"][i])

        plot_grid(
            history=[tuple(row) for row in history],
            labels=train_labels,
            n_cols=1,
            save_path=save_path,
            x_label="Iteration",
        )

        n_test = len(self.test_metrics)
        test_labels = ["Loss", "Accuracy"]
        test_history = [
            [self.test_metrics["loss"][i], self.test_metrics["accuracy"][i]]
            for i in range(n_test)
        ]

        if self.compute_top5:
            test_labels.append("Top-5 Accuracy")

            for i in range(n_test):
                test_history[i].append(self.test_metrics["top5_accuracy"][i])

        plot_grid(
            history=[tuple(row) for row in test_history],
            labels=test_labels,
            n_cols=1,
            save_path=save_path,
            x_label="Iteration",
            save_title="Test Metrics",
        )

        if save_path:
            with open(os.path.join(save_path, "train_params.txt"), "w") as f:
                f.write(f"epochs: {self.epochs}\n")
                f.write(f"lr: {self.lr}\n")
                f.write(f"min_workers: {self.min_workers}\n")
                f.write(f"shard_size: {self.shard_size}\n")
                f.write(f"batch_size: {self.batch_size}\n")
                f.write(f"max_staleness: {self.max_staleness}\n")
                f.write(f"test_each: {self.test_each}\n")
                f.write(f"run_epochs: {self._scheduler.current_epoch}\n")
                f.write(f"k: {self.k}\n")

    def run(self, host: str = "0.0.0.0", port: int = 9999) -> None:
        self.start_server(host=host, port=port)
        log.info(f"Servidor async listo en {host}:{port}")

        try:
            self.train()
        except KeyboardInterrupt:
            log.info("Entrenamiento interrumpido por usuario")
        finally:
            self.stop_server()

    def stop_server(self) -> None:
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

            try:
                self._wait_and_register_workers()
                self._broadcast_fast(DDPMessage.msg("stop"))
                self._broadcast_fast(DDPMessage.msg("metrics"))
            except Exception as e:
                log.error(f"Error al obtener metricas: {e}")

        self.results()
        super().stop_server()
