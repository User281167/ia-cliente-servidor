import threading
import time
from typing import Optional

import numpy as np
import pandas as pd

from async_impl import AsyncGradServer
from ddp.logger import log


class RennalaSGDServer(AsyncGradServer):
    def __init__(self, B: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.B = B
        self._accumulated_gradients: Optional[dict] = None
        self._accumulated_gradients_count = 0
        self._accept_lock = threading.Lock()
        self.max_staleness = 0
        self._last_test_k = 0

        self.metrics = pd.DataFrame(
            columns=[
                "loss",
                "accuracy",
                "top5_accuracy",
                "grad_norm",
                "elapsed",
            ]
        )

    def _register_event_handlers(self) -> None:
        super()._register_event_handlers()

        @self.on("result")
        def _handle_result(msg: dict) -> None:
            if self._stop_event.is_set():
                return

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
            grad_norm = float("nan")

            with self._accept_lock:
                k_now = self.k
                staleness = k_now - iter_sent

                if staleness == 0:
                    if self._accumulated_gradients is None:
                        self._accumulated_gradients = {
                            k: v.copy() for k, v in grads.items()
                        }
                    else:
                        for k, v in grads.items():
                            self._accumulated_gradients[k] += v

                    self._accumulated_gradients_count += 1

                    if self._accumulated_gradients_count == self.B:
                        avg_grads = {
                            k: v / self.B
                            for k, v in self._accumulated_gradients.items()
                        }

                        grad_norm = self._apply_gradient(avg_grads, self.lr)
                        self._accumulated_gradients = None
                        self._accumulated_gradients_count = 0
                        self.k += 1

                fresh_state = self.get_weights()
                k_new = self.k

                send_test = k_new % self.test_each == 0 and k_new != self._last_test_k

                if send_test:
                    self._last_test_k = k_new

                if staleness == 0:
                    grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

                    self.metrics.loc[len(self.metrics)] = [
                        loss,
                        accuracy,
                        top5_accuracy,
                        grad_norm,
                        time.perf_counter() - t0,
                    ]

            if send_test:
                self._send_test(wid, fresh_state, k_new)
            else:
                self._send_step_to(wid, fresh_state, k_new)

            if shard_idx is not None:
                self._scheduler.complete(wid, shard_idx)

            if staleness == 0:
                txt = (
                    f"[k={k_now}] epoch={self._scheduler.current_epoch}/{self.epochs}"
                    f" | worker={wid} "
                    f" | samples={samples} | loss={loss:.4f} | accuracy={accuracy:.4f} "
                    f" | grad_norm={grad_norm:.4f}"
                )

                if self.compute_top5:
                    txt += f" | top5={top5_accuracy:.4f} "

                log.info(txt)
            else:
                log.info(
                    f"[k={k_now}] worker={wid} staleness={staleness} resultado descartado"
                )
