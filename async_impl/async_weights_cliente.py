import time

import numpy as np
import torch

from ddp.pickle_utils import log, send_msg

from .async_grads_cliente import AsyncGradWorker


class AsyncWeightsWorker(AsyncGradWorker):
    """
    Worker async que recibe pesos, entrena minibatches locales y devuelve
    delta = w_local - w_global.
    """

    def get_delta(self, w_local, w_global):
        return {
            k: (w_local[k] - w_global[k]).detach().cpu().numpy().astype(np.float32)
            for k in w_global
        }

    def train(self, t0, w_global):
        self._ensure_loaders()
        self.model.train()

        total_loss = torch.tensor(0.0)
        total_correct = torch.tensor(0.0)
        total_correct_top5 = torch.tensor(0.0)
        total_samples = torch.tensor(0.0)
        n_batches = 0

        indices = self.sampler.get_indices(self.dataset, self.assignment)
        self.indexed_dataset.set_indices(indices)

        for X, y in self.loader:
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

            if self.compute_top5:
                _, pred5 = torch.topk(outputs, 5, dim=1)
                total_correct_top5 += pred5.eq(y.view(-1, 1)).sum().item()

            if n_batches % 10 == 0:
                print(
                    f"Batch {n_batches}",
                    f"| loss: {loss.item():.4f}",
                    f"| acc: {float(total_correct / total_samples):.4f}",
                    f"| top5 acc: {float(total_correct_top5 / total_samples):.4f}"
                    if self.compute_top5
                    else "",
                    f"| size: {y.size(0)}",
                    end="\r",
                )

        if self.scheduler is not None:
            self.scheduler.step()

        w_local = self.model.state_dict()
        delta = self.get_delta(w_local, w_global)

        if total_samples.item() == 0:
            self._last_train_top5_accuracy = 0.0
            return delta, 0.0, 0.0, 0.0, 0.0, 0

        self._last_train_top5_accuracy = (total_correct_top5 / total_samples).item()
        avg_acc = total_correct / total_samples
        avg_loss = total_loss / total_samples
        elapse = time.perf_counter() - t0
        throughput = total_samples / elapse if elapse > 0 else torch.tensor(0.0)

        return (
            delta,
            avg_loss.item(),
            avg_acc.item(),
            elapse,
            throughput.item(),
            int(total_samples.item()),
        )

    def _register_handlers(self):
        super()._register_handlers()

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
            w_global = {k: v.clone() for k, v in self.model.state_dict().items()}

            t0 = time.perf_counter()
            delta, loss, accuracy, elapse, throughput, samples = self.train(
                t0,
                w_global,
            )

            top5_accuracy = getattr(self, "_last_train_top5_accuracy", 0.0)

            txt = (
                f"Worker: {self._worker_id} | epoch={epoch} "
                f"| acc={accuracy:.4f} | loss={loss:.4f} "
                f"| elapsed={elapse:.4f} | throughput={throughput:.4f}"
            )

            if self.compute_top5:
                txt += f"| top5 acc={top5_accuracy:.4f}"

            log.info(txt)

            self.metrics.loc[len(self.metrics)] = [
                loss,
                accuracy,
                top5_accuracy,
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
                        "samples": samples,
                        "loss": loss,
                        "accuracy": accuracy,
                        "top5_accuracy": top5_accuracy,
                        "iter_sent": k_iter,
                        "shard_idx": self.assignment.shard_idx,
                    },
                },
            )
