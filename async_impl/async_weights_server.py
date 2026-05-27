import os
import time

import pandas as pd
import torch
from ddp.logger import log
from utils import plot_grid

from .async_grads_server import AsyncGradServer


class AsyncWeightsServer(AsyncGradServer):
    """
    Servidor async que aplica delta de pesos con staleness.
    Hereda comunicacion/scheduler/evaluacion de AsyncGradServer.
    """

    def __init__(
        self,
        data_len: int,
        epochs: int = 20,
        lr: float = 0.001,
        gamma: float = 0.1,
        shard_size: int = 5000,
        batch_size: int = 128,
        max_staleness: int = 10,
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

        super().__init__(
            data_len=data_len,
            epochs=epochs,
            lr=lr,
            shard_size=shard_size,
            batch_size=batch_size,
            max_staleness=max_staleness,
            min_workers=min_workers,
            config=config,
            save_path=save_path,
        )

        self.gamma = gamma
        self.metrics = pd.DataFrame(
            columns=[
                "loss",
                "accuracy",
                "eval_loss",
                "eval_accuracy",
                "delta_norm",
                "staleness",
                "gamma",
                "elapsed",
            ]
        )

    def _gamma(self, staleness: int) -> float:
        return self.gamma / (1.0 + staleness)

    def _apply_delta(self, delta: dict, gamma: float) -> float:
        state = self.model.state_dict()
        delta_norm_sq = 0.0

        for name, value in delta.items():
            if name not in state:
                continue

            d_t = torch.as_tensor(
                value,
                dtype=state[name].dtype,
                device=state[name].device,
            )
            state[name] = state[name] + gamma * d_t
            delta_norm_sq += torch.linalg.vector_norm(d_t.float()).item() ** 2

        self.model.load_state_dict(state)
        return delta_norm_sq**0.5

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

            delta = payload["delta"]
            samples = payload.get("samples", 0)
            loss = payload.get("loss", float("nan"))
            accuracy = payload.get("accuracy", float("nan"))
            eval_loss = payload.get("eval_loss", float("nan"))
            eval_accuracy = payload.get("eval_accuracy", float("nan"))
            iter_sent = payload.get("iter_sent", self.k)
            shard_idx = payload.get("shard_idx", None)

            with self._k_lock:
                k_now = self.k
                staleness = k_now - iter_sent
                gamma = self._gamma(staleness)

                if staleness <= self.max_staleness:
                    delta_norm = self._apply_delta(delta, gamma)
                else:
                    delta_norm = float("nan")

                self.k += 1
                fresh_state = self._get_state_numpy()
                k_new = self.k

                if staleness <= self.max_staleness:
                    self.metrics.loc[len(self.metrics)] = [
                        loss,
                        accuracy,
                        eval_loss,
                        eval_accuracy,
                        delta_norm,
                        staleness,
                        gamma,
                        time.perf_counter() - t0,
                    ]

            self._send_step_to(wid, fresh_state, k_new)

            if shard_idx is not None:
                self._scheduler.complete(wid, shard_idx)

            if k_now % 10 == 0 and staleness <= self.max_staleness:
                log.info(
                    f"[k={k_now}] epoch={self._scheduler.current_epoch}/{self.epochs} "
                    f"worker={wid} staleness={staleness} gamma={gamma:.6f} "
                    f"samples={samples} loss={loss:.4f} eval_loss={eval_loss:.4f} "
                    f"accuracy={accuracy:.4f} eval_accuracy={eval_accuracy:.4f} "
                    f"delta_norm={delta_norm:.4f}"
                )

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

    def results(self) -> None:
        save_path = self.save_path

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.metrics.to_excel(os.path.join(save_path, "metrics_server.xlsx"))
            self.metrics.describe(percentiles=[0.1, 0.5, 0.9]).to_excel(
                os.path.join(save_path, "description_server.xlsx"), index=True
            )

        if len(self.metrics) > 0:
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

        if save_path:
            with open(os.path.join(save_path, "train_params.txt"), "w") as f:
                f.write(f"epochs: {self.epochs}\n")
                f.write(f"lr: {self.lr}\n")
                f.write(f"gamma: {self.gamma}\n")
                f.write(f"min_workers: {self.min_workers}\n")
                f.write(f"shard_size: {self.shard_size}\n")
                f.write(f"batch_size: {self.batch_size}\n")
                f.write(f"max_staleness: {self.max_staleness}\n")
                f.write(f"run_epochs: {self._scheduler.current_epoch}\n")
                f.write(f"k: {self.k}\n")
