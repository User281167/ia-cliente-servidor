import time

import numpy as np
import torch

from ddp.pickle_utils import log, send_msg
from sync import SyncGradWorker


class SyncWeightsWorker(SyncGradWorker):
    """
    Cliente worker para el entrenamiento distribuido con sincronización de pesos.
    """

    def __init__(self, host, port, save_path):
        super().__init__(host, port, save_path)

    def train(self, seed, t0, w_global=None):
        total_loss = torch.tensor(0.0)
        total_correct = torch.tensor(0.0)
        total_correct_top5 = torch.tensor(0.0)
        total_samples = torch.tensor(0.0)
        total_correct_top5 = 0
        n_batches = 0

        for X, y in self.loader:
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
                    end="\r",
                )

        if self.scheduler is not None:
            self.scheduler.step()

        # Δw = w_local - w_global
        w_local = self.model.state_dict()
        delta = {
            k: (w_local[k] - w_global[k]).cpu().numpy().astype(np.float32)
            for k in w_global
        }

        self._last_train_top5_accuracy = (total_correct_top5 / total_samples).item()
        avg_acc = total_correct / total_samples
        avg_loss = total_loss / total_samples

        elapse = time.perf_counter() - t0
        throughput = total_samples / elapse

        return (
            delta,
            avg_loss.item(),
            avg_acc.item(),
            elapse,
            throughput.item(),
            total_samples.item(),
        )

    def _register_handlers(self):
        """
        Registra los manejadores de mensajes del servidor.
        Ejecuta las funciones correspondientes cuando se reciben mensajes del servidor.
        """
        super()._register_handlers()

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
            w_global = {k: v.clone() for k, v in self.model.state_dict().items()}

            self.test_sampler.set_epoch(seed, self.rank, self.world_size)
            self.sampler.set_epoch(seed, self.rank, self.world_size)

            eval_loss, eval_correct, eval_total = self.test(seed)
            delta, avg_loss, avg_acc, elapse, throughput, total_samples = self.train(
                seed, t0, w_global
            )

            top5_accuracy = getattr(self, "_last_train_top5_accuracy", 0.0)
            eval_top5_accuracy = getattr(self, "_last_test_top5_accuracy", 0.0)

            msg = (
                f"Worker {self.rank}: epoch={epoch} | "
                f"acc={avg_acc:.4f} | loss={avg_loss:.4f} | "
                f"elapse={elapse:.4f} | throughput={throughput:.4f}"
            )

            if self.compute_top5:
                msg += f"| top5 acc = {top5_accuracy:.4f}"
                msg += f"| eval top5 acc = {eval_top5_accuracy:.4f}"

            log.info(msg)

            self.metrics.loc[len(self.metrics)] = [
                avg_loss,
                avg_acc,
                top5_accuracy,
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
                        "top5_accuracy": top5_accuracy,
                        # test
                        "eval_loss": eval_loss,  # suma, no promedio
                        "eval_correct": eval_correct,
                        "eval_total": eval_total,
                        "eval_top5_correct": int(eval_top5_accuracy * eval_total)
                        if eval_total > 0
                        else 0,
                    },
                },
            )
