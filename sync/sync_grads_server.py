import os
import time

import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import MulticlassConfusionMatrix

from ddp import DDPServer
from ddp.logger import log
from ddp.message import DDPMessage
from ddp.pickle_utils import send_msg
from utils import format_elapse, plot_grid, time_wrapper


class SyncGradServer(DDPServer):
    """
    Servidor para el entrenamiento distribuido con promedio de gradientes.
    """

    def __init__(
        self,
        epochs: int = 20,
        lr: float = 0.001,
        batch_size: int = 128,
        min_workers: int = 1,
        config: dict | None = None,
        save_path: str | None = None,
        load_model: bool = False,
    ):
        if config is None:
            config = {
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
            }

        super().__init__(min_workers, config)
        config.setdefault("top5", False)
        self.compute_top5 = config["top5"]
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.criterion = self.optimizer = self.scheduler = None

        self.epochs = epochs
        self.current_epoch = 0

        self.test_loader = None
        self.save_path = save_path
        self.num_classes = 10

        self.metrics = pd.DataFrame(
            columns=[
                "workers",
                "worker_res",
                "loss",
                "accuracy",
                "top5_accuracy",
                "eval_loss",
                "eval_accuracy",
                "eval_top5_accuracy",
                "grad_norm",
                "elapsed",
            ]
        )

    def results(self):
        """Guarda las métricas en un archivo Excel y genera gráfico de resultados."""
        save_path = self.save_path

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        if save_path:
            self.metrics.to_excel(os.path.join(save_path, "metrics_server.xlsx"))

            description = self.metrics.describe(percentiles=[0.1, 0.5, 0.9])
            description.to_excel(
                os.path.join(save_path, "description_server.xlsx"),
                index=True,
            )

        n = len(self.metrics)
        history = []

        for i in range(n):
            row = [
                (self.metrics["loss"][i], self.metrics["eval_loss"][i]),
                (self.metrics["accuracy"][i], self.metrics["eval_accuracy"][i]),
            ]

            if self.compute_top5:
                row.append(
                    (
                        self.metrics["top5_accuracy"][i],
                        self.metrics["eval_top5_accuracy"][i],
                    )
                )

            row.append(self.metrics["grad_norm"][i])
            history.append(tuple(row))

        labels = [("Loss", "Train", "Test"), ("Accuracy", "Train", "Test")]

        if self.compute_top5:
            labels.append(("Top-5 Accuracy", "Train", "Test"))

        labels.append("Grad Norm")

        plot_grid(
            history=history,
            labels=labels,
            n_cols=1,
            save_path=save_path,
        )

        # evaluate classification
        acc, _ = self.evaluate_classification()

        # argumentos
        if save_path:
            with open(os.path.join(save_path, "train_params.txt"), "w") as f:
                f.write(f"epochs: {self.epochs}\n")
                f.write(f"lr: {self.lr}\n")
                f.write(f"min_workers: {self.min_workers}\n")
                f.write(f"batch_size: {self.batch_size}\n")
                f.write(f"Final accuracy: {acc}")

    def evaluate(self) -> tuple[float, float]:
        """Evalua el modelo en el test set y devuelve loss y accuracy."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():  # no se actualizan los pesos
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return total_loss / total, correct / total

    def evaluate_classification(self) -> tuple[float, torch.Tensor]:
        """Evalua accuracy y confusion matrix en el test set."""
        self.model.eval()
        correct = 0
        total = 0
        confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes).to(
            self.device
        )

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                confusion_matrix.update(predicted, labels)

        return correct / total, confusion_matrix.compute().cpu()

    def _send_assign(self, n_workers: int, epoch: int):
        """
        Envía el mensaje de asignación a todos los workers.
        Cada worker recibe un mensaje con su ID, rank, world_size y epoch antes de comenzar el entrenamiento.
        """
        items = self._get_registered_workers()

        for i, (wid, sock) in enumerate(items):
            msg = DDPMessage.assign(
                worker_id=wid,
                rank=i,
                world_size=n_workers,
                epoch=epoch,
            )

            self._assignments[wid] = msg

            try:
                send_msg(sock, msg)
            except Exception as e:
                log.warning(f"Worker {wid} fallo assign: {e}")

    def _aggregate(self, results):
        """
        Promedio ponderado de los gradientes de los workers.

        Args:
            results: [{delta, samples}]
        """
        N_total = sum(r["samples"] for r in results)
        accum_grads: dict[str, torch.Tensor] = {}

        for i in range(len(results)):
            n_i = results[i]["samples"]
            grad = n_i / N_total
            grad_i = results[i]["grads"]

            for k, d in grad_i.items():
                d = torch.as_tensor(d, device=self.device)

                if k not in accum_grads:
                    accum_grads[k] = grad * d
                else:
                    accum_grads[k] += grad * d

        # Norma L2
        # Permite saber cuanto se está moviendo el gradiente en cada iteración
        # Permite observar desvanecimiento o explotación del gradiente
        gnorm = 0.0

        # Aplicar gradientes
        # promedio de gradientes
        # actualizar modelo de pytorch
        for name, param in self.model.named_parameters():
            g = accum_grads[name]
            param.grad = g.detach()
            gnorm += (g**2).sum().item()

        gnorm = gnorm**0.5

        # optimizar
        self.optimizer.step()
        self.optimizer.zero_grad()

        return gnorm

    def _aggregate_metrics(self, results):
        """
        Agrega las métricas de los resultados de test de los workers.
        """
        N = sum(r["samples"] for r in results)
        loss = sum(r["loss"] * r["samples"] for r in results) / N
        accuracy = sum(r["accuracy"] * r["samples"] for r in results) / N
        top5_accuracy = sum(r["top5_accuracy"] * r["samples"] for r in results) / N

        # eval distribuido — sumar counts crudos, no promediar
        eval_total = sum(r["eval_total"] for r in results)
        eval_loss = sum(r["eval_loss"] for r in results) / eval_total
        eval_accuracy = sum(r["eval_correct"] for r in results) / eval_total
        eval_top5_accuracy = (
            sum(r.get("eval_top5_correct", 0) for r in results) / eval_total
            if eval_total > 0
            else 0.0
        )

        return (
            loss,
            accuracy,
            top5_accuracy,
            eval_loss,
            eval_accuracy,
            eval_top5_accuracy,
        )

    def step(self):
        """
        Ejecuta un paso de entrenamiento distribuido.
        Espera a que los workers estén listos, envía los pesos actuales,
        luego envía el mensaje de step y recopila los resultados.
        """
        n_workers = self._wait_and_register_workers()

        if n_workers is None:
            log.warning("Timeout esperando workers (saltar época)")
            return

        t0 = time.perf_counter()
        self.current_workers = n_workers

        # pesos y parámetros del modelo en pytorch
        state = {
            k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in self.model.state_dict().items()
        }

        self._broadcast_weights(state)
        self._send_assign(n_workers, self.current_epoch)
        self._broadcast_step(self.current_epoch)
        results = self._collect_results()

        if not results:
            log.warning("No se recibieron resultados, saltando época")
            return

        # result {type, payload} obtener solo el payload
        results = [r["payload"] for r in results]

        loss, accuracy, top5_accuracy, eval_loss, eval_accuracy, eval_top5_accuracy = (
            self._aggregate_metrics(results)
        )
        gnorm = self._aggregate(results)

        elapsed = time.perf_counter() - t0

        self.metrics.loc[self.current_epoch] = [
            self.current_workers,
            len(results),
            loss,
            accuracy,
            top5_accuracy,
            eval_loss,
            eval_accuracy,
            eval_top5_accuracy,
            gnorm,
            elapsed,
        ]

        msg = (
            f"Epoch {self.current_epoch + 1}/{self.epochs} "
            f"| loss: {loss:.4f} | eval_loss: {eval_loss:.4f} "
            f"| accuracy: {accuracy:.4f} | eval_accuracy: {eval_accuracy:.4f} "
            f"| norm: {gnorm:.4f} | elapsed: {format_elapse(elapsed)}"
        )

        if self.compute_top5:
            msg += f" top5 acc: {top5_accuracy:.4f} |"
            msg += f" test top5 acc: {eval_top5_accuracy:.4f}"

        log.info(msg)

        self.current_epoch += 1

    @time_wrapper
    def train(self):
        """Entrena el modelo durante el número de épocas especificado."""
        while self.current_epoch < self.epochs:
            self.step()

    def close(self):
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

            self._wait_and_register_workers()
            self._broadcast_fast({"type": "metrics"})
            results = self._collect("metrics")

            for result in results:
                payload = result["payload"]
                df = pd.DataFrame(payload["data_frame"])
                rank = payload["rank"]
                description = df.describe()

                df.to_excel(os.path.join(self.save_path, f"metrics_{rank}.xlsx"))

                description = self.metrics.describe(percentiles=[0.1, 0.5, 0.9])
                description.to_excel(
                    os.path.join(self.save_path, f"description_{rank}.xlsx"),
                    index=True,
                )

            self.results()

            state_dict = self.model.state_dict()
            path = os.path.join(self.save_path, "model.pth")
            torch.save(state_dict, path)

        self.stop_server()

    def run(self, host: str = "0.0.0.0", port: int = 9999):
        """
        Inicia el servidor y entrena el modelo.

        Finalizar pedir métricas y guardarlas en Excel.
        """
        self.start_server(host=host, port=port)

        try:
            self.train()
        except KeyboardInterrupt:
            log.info("Entrenamiento interrumpido por el usuario")
        finally:
            self.close()
