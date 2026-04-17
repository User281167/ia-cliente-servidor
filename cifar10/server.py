import os
import time

import numpy as np
import pandas as pd
import torch
from torchinfo import summary
from torchmetrics.classification import MulticlassConfusionMatrix

from ddp import DDPServer
from ddp.logger import log
from ddp.message import DDPMessage
from ddp.pickle_utils import send_msg
from utils import format_elapse, plot_confusion_matrix, plot_grid, time_wrapper

from .load_data import cifar10_classes, get_cifar10_dataloader
from .model import Cifar10Model


class CIFAR10Server(DDPServer):
    """
    Servidor para el entrenamiento distribuido del modelo CIFAR-10.
    """

    def __init__(
        self,
        gray: bool = False,
        normalize: bool = False,
        conv: bool = False,
        epochs: int = 20,
        lr: float = 0.001,
        batch_size: int = 128,
        min_workers: int = 1,
    ):
        super().__init__(min_workers)
        self.gray = gray
        self.normalize = normalize
        self.conv = conv
        self.lr = lr
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Cifar10Model(gray=gray, conv=conv).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        summary(self.model, input_size=(1, 1 if gray else 3, 32, 32))

        self.epochs = epochs
        self.current_epoch = 0

        self.test_loader = get_cifar10_dataloader(
            train=False, gray=gray, normalize=normalize
        )

        self.metrics = pd.DataFrame(
            columns=[
                "loss",
                "accuracy",
                "eval_loss",
                "eval_accuracy",
                "grad_norm",
                "elapsed",
            ]
        )

    def results(self, save_path: str | None):
        """Guarda las métricas en un archivo Excel y genera gráfico de resultados."""

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        if save_path:
            self.metrics.to_excel(os.path.join(save_path, "metrics_server.xlsx"))

            description = self.metrics.describe(percentiles=[0.1, 0.5, 0.9])
            description.to_excel(
                os.path.join(save_path, "description_server.xlsx"),
                index=True,
            )

        plot_grid(
            history=[
                (
                    # unir train/test en una sola gráfica
                    (self.metrics["loss"][i], self.metrics["eval_loss"][i]),
                    ((self.metrics["accuracy"][i], self.metrics["eval_accuracy"][i])),
                    self.metrics["grad_norm"][i],
                )
                for i in range(len(self.metrics))
            ],
            labels=[
                ("Loss", "Train", "Test"),
                ("Accuracy", "Train", "Test"),
                "Grad Norm",
            ],
            n_cols=1,
            save_path=save_path,
        )

        # evaluate classification
        acc, conf = self.evaluate_classification()
        plot_confusion_matrix(conf, save_path=save_path, class_names=cifar10_classes)

        # argumentos
        if save_path:
            with open(os.path.join(save_path, "train_params.txt"), "w") as f:
                f.write(f"epochs: {self.epochs}\n")
                f.write(f"lr: {self.lr}\n")
                f.write(f"workers: {self.workers}\n")
                f.write(f"min_workers: {self.min_workers}\n")
                f.write(f"gray: {self.gray}\n")
                f.write(f"normalize: {self.normalize}\n")
                f.write(f"conv: {self.conv}\n")
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
        confusion_matrix = MulticlassConfusionMatrix(num_classes=10).to(self.device)

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
        with self._workers_lock:
            items = list(self._workers.items())

        for i, (wid, sock) in enumerate(items):
            msg = DDPMessage.assign(
                worker_id=wid,
                rank=i,
                world_size=n_workers,
                epoch=epoch,
                batch_size=self.batch_size,
            )

            self._assignments[wid] = msg

            try:
                send_msg(sock, msg)
            except Exception as e:
                log.warning(f"Worker {wid} fallo assign: {e}")

    def step(self):
        """
        Ejecuta un paso de entrenamiento distribuido.
        Espera a que los workers estén listos, envía los pesos actuales,
        luego envía el mensaje de step y recopila los resultados.
        """
        t0 = time.perf_counter()
        n_workers = self._wait_workers()

        if n_workers is None:
            log.warning("Timeout esperando workers (saltar época)")
            return

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

        accum_grads: dict[str, torch.Tensor] = {}

        # acomular gradientes
        for msg in results:
            grads = msg["payload"]["grads"]

            for k, g in grads.items():
                g = torch.as_tensor(g)

                if k not in accum_grads:
                    accum_grads[k] = g.clone()
                else:
                    accum_grads[k] += g

        # promedio
        for k in accum_grads:
            accum_grads[k] /= len(results)

        # Norma L2
        # Permite saber cuanto se está moviendo el gradiente en cada iteración
        # Permite observar desvanecimiento o explotación del gradiente
        gnorm = 0.0

        # Aplicar gradientes
        # actualizar modelo de pytorch
        for name, param in self.model.named_parameters():
            g = accum_grads[name] / len(results)
            param.grad = g.detach()
            gnorm += (g**2).sum().item()

        gnorm = gnorm**0.5

        # optimizar
        self.optimizer.step()
        self.optimizer.zero_grad()

        elapsed = time.perf_counter() - t0
        loss = float(sum(r["payload"]["loss"] for r in results) / len(results))
        accuracy = float(sum(r["payload"]["accuracy"] for r in results) / len(results))

        eval_loss, eval_accuracy = self.evaluate()

        self.metrics.loc[self.current_epoch] = [
            loss,
            accuracy,
            eval_loss,
            eval_accuracy,
            gnorm,
            elapsed,
        ]

        log.info(
            f"Epoch {self.current_epoch + 1}/{self.epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f} - eval_loss: {eval_loss:.4f} - eval_accuracy: {eval_accuracy:.4f} - gnorm: {gnorm:.4f} - elapsed: {format_elapse(elapsed)}"
        )

        self.current_epoch += 1

    @time_wrapper
    def train(self):
        """Entrena el modelo durante el número de épocas especificado."""
        while self.current_epoch < self.epochs:
            self.step()

    def run(self, host: str = "0.0.0.0", port: int = 9999):
        """Inicia el servidor y entrena el modelo."""
        self.start_server(host=host, port=port)

        try:
            self.train()
        finally:
            self.stop_server()
