import os
import threading
import time

import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import MulticlassConfusionMatrix

from cifar10.load_data import cifar10_classes, cifar10_data_len, get_cifar10_dataloader
from cifar10.model import cifar10_get_model
from ddp import DDPAsyncServer
from ddp.logger import log
from ddp.message import DDPMessage
from ddp.pickle_utils import send_msg
from ddp.shard_scheduler import ShardScheduler
from utils import plot_confusion_matrix, plot_grid, time_wrapper


class CIFAR10Server(DDPAsyncServer):
    """
    Servidor para entrenamiento async con delta de pesos.

    Flujo:
      - Worker recibe pesos actuales + assignment.
      - Worker entrena localmente y devuelve delta = w_local - w_global.
      - Servidor aplica w <- w + gamma * delta con correccion por staleness.
      - Servidor envia de inmediato nuevos pesos + siguiente assignment.
    """

    def __init__(
        self,
        gray: bool = False,
        normalize: bool = False,
        conv: bool = False,
        epochs: int = 20,
        lr: float = 0.001,
        gamma: float = 0.1,
        shard_size: int = 5000,
        batch_size: int = 128,
        max_staleness: int = 10,
        save_path: str | None = None,
    ):
        config = {
            "gray": gray,
            "normalize": normalize,
            "conv": conv,
            "epochs": epochs,
            "lr": lr,
        }

        super().__init__(worker_config=config)

        data_len = cifar10_data_len()
        self.gray = gray
        self.normalize = normalize
        self.conv = conv
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.shard_size = shard_size
        self.save_path = save_path
        self.epochs = epochs
        self._scheduler = ShardScheduler(data_len, shard_size, batch_size)
        self.max_staleness = max_staleness

        self.k = 0
        self._k_lock = threading.Lock()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.criterion, self.optimizer = cifar10_get_model(
            gray=gray, conv=conv, lr=lr, device=self.device
        )

        self.test_loader = get_cifar10_dataloader(
            train=False, gray=gray, normalize=normalize
        )

        self.metrics = pd.DataFrame(
            columns=["loss", "eval_loss", "accuracy", "eval_accuracy", "delta_norm"]
        )

        self._register_event_handlers()

    def _remove_dead(self, wids: list[int]) -> None:
        for wid in wids:
            self._scheduler.requeue(wid)

        super()._remove_dead(wids)

    def _gamma(self, staleness: int) -> float:
        return self.gamma / (1.0 + staleness)

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

    def evaluate_classification(self) -> tuple[float, torch.Tensor]:
        self.model.eval()
        correct, total = 0, 0
        confusion_matrix = MulticlassConfusionMatrix(num_classes=10).to(self.device)

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

    def _apply_delta(self, delta: dict, gamma: float) -> float:
        """Aplica el delta de pesos al modelo.

        return:
            La norma del delta de pesos.
        """
        state = self.model.state_dict()
        delta_norm_sq = 0.0

        for name, d in delta.items():
            if name not in state:
                continue

            d_t = torch.as_tensor(d, dtype=state[name].dtype, device=state[name].device)
            state[name] = state[name] + gamma * d_t
            delta_norm_sq += d_t.norm().item() ** 2

        self.model.load_state_dict(state)
        return delta_norm_sq**0.5

    def _send_step_to(self, wid: int, state: dict, k: int) -> None:
        with self._workers_lock:
            sock = self._workers.get(wid)

        if sock is None:
            log.error(f"Worker {wid} ya no esta conectado, skip send")
            return

        assignment = self._scheduler.next_shard(wid)
        current_epoch = assignment.epoch

        try:
            send_msg(
                sock,
                DDPMessage.msg(
                    "step",
                    iter=k,
                    epoch=current_epoch,
                    weights=state,
                    assignment=assignment,
                ),
            )
        except Exception as e:
            log.error(f"No se pudo enviar paso al worker {wid}: {e}")
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
            shard_idx = payload.get("shard_idx", None)
            delta = payload["delta"]
            acc_value = payload.get("acc", float("nan"))
            loss_value = payload.get("loss", float("nan"))
            iter_sent = payload.get("iter_sent", self.k)

            test_acc = payload.get("test_acc", float("nan"))
            test_loss = payload.get("test_loss", float("nan"))

            with self._k_lock:
                k_now = self.k
                staleness = k_now - iter_sent

                if staleness <= self.max_staleness:
                    gamma = self._gamma(staleness)

                    delta_norm = self._apply_delta(delta, gamma)
                    self.metrics.loc[len(self.metrics)] = [
                        loss_value,
                        test_loss,
                        acc_value,
                        test_acc,
                        delta_norm,
                    ]
                else:
                    delta_norm = float("nan")
                    gamma = self._gamma(staleness)

                self.k += 1
                fresh_state = self._get_state_numpy()
                k_new = self.k

            self._send_step_to(wid, fresh_state, k_new)
            current_epoch = self._scheduler.current_epoch

            if k_now % 10 == 0 and staleness <= self.max_staleness:
                log.info(
                    f"[k={k_now}] epoch={current_epoch}/{self.epochs} "
                    f"worker={wid} d={staleness} gamma={gamma:.6f} "
                    f"loss={loss_value:.4f} test_loss={test_loss:.4f} "
                    f"accuracy={acc_value:.4f} test_acc={test_acc:.4f} "
                    f"delta_norm={delta_norm:.4f}"
                )

            if shard_idx is not None:
                self._scheduler.complete(wid, shard_idx)

        @self.on("metrics")
        def _handle_metrics(msg: dict) -> None:
            if self.save_path is None:
                return

            wid = msg["worker_id"]
            payload = msg["payload"]
            df = pd.DataFrame(payload["data_frame"])
            description = df.describe()

            df.to_excel(os.path.join(self.save_path, f"metrics_{wid}.xlsx"))

            description = self.metrics.describe(percentiles=[0.1, 0.5, 0.9])
            description.to_excel(
                os.path.join(self.save_path, f"description_{wid}.xlsx"),
                index=True,
            )

    @time_wrapper
    def train(self) -> None:
        while self._scheduler.current_epoch <= self.epochs:
            time.sleep(1)

        log.info(f"Entrenamiento completado - k={self.k} iteraciones totales")

    def results(self) -> None:
        save_path = self.save_path

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.metrics.to_excel(os.path.join(save_path, "metrics_server.xlsx"))
            self.metrics.describe(percentiles=[0.1, 0.5, 0.9]).to_excel(
                os.path.join(save_path, "description_server.xlsx"), index=True
            )

        plot_grid(
            history=[
                (
                    (self.metrics["loss"][i], self.metrics["eval_loss"][i]),
                    (
                        self.metrics["accuracy"][i],
                        self.metrics["eval_accuracy"][i],
                    ),
                    self.metrics["delta_norm"][i],
                )
                for i in range(len(self.metrics))
            ],
            labels=[
                ("Loss", "Train", "Test"),
                ("Accuracy", "Train", "Test"),
                "Delta Norm",
            ],
            n_cols=1,
            save_path=save_path,
            x_label="Iteration",
        )

        acc, conf = self.evaluate_classification()
        plot_confusion_matrix(conf, save_path=save_path, class_names=cifar10_classes)

        if save_path:
            with open(os.path.join(save_path, "train_params.txt"), "w") as f:
                f.write(f"epochs: {self.epochs}\n")
                f.write(f"lr: {self.lr}\n")
                f.write(f"min_workers: {self.min_workers}\n")
                f.write(f"gray: {self.gray}\n")
                f.write(f"normalize: {self.normalize}\n")
                f.write(f"conv: {self.conv}\n")
                f.write(f"shard_size: {self.shard_size}\n")
                f.write(f"batch_size: {self.batch_size}\n")
                f.write(f"Final accuracy: {acc}\n")
                f.write(f"Run epochs: {self._scheduler.current_epoch}")

    def run(self, host: str = "0.0.0.0", port: int = 9999) -> None:
        self.start_server(host=host, port=port)
        log.info("Servidor listo. Esperando workers...")

        try:
            self.train()
        except KeyboardInterrupt:
            log.info("Interrumpido por usuario")
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
