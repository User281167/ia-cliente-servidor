import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassConfusionMatrix

from ddp import DDPServer
from ddp.logger import log
from ddp.message import DDPMessage
from ddp.pickle_utils import send_msg
from utils import format_elapse, plot_grid, time_wrapper

from .load_data import ShardSampler, TinyImageNetLazy
from .model import get_tiny_imagenet_model
from .report import excel_report


class TinyImageNetServer(DDPServer):
    """
    Servidor para el entrenamiento distribuido del modelo Tiny ImageNet.
    """

    def __init__(
        self,
        epochs: int = 20,
        lr: float = 0.001,
        batch_size: int = 128,
        min_workers: int = 1,
        save_path: str | None = None,
        worker_timeout: int = 60 * 5,
    ):
        config = {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
        }

        super().__init__(min_workers, config)
        self.lr = lr
        self.batch_size = batch_size
        self.save_path = save_path
        self.current_workers = 0
        self.WORKER_TIMEOUT = worker_timeout

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {self.device}")

        self.model, self.criterion, _, _ = get_tiny_imagenet_model(
            lr=lr, epochs=epochs, device=self.device
        )

        self.epochs = epochs
        self.current_epoch = 0
        self.loader = None

        self.load_test_data()

        self.metrics = pd.DataFrame(
            columns=[
                "workers",
                "worker_res",
                "loss",
                "accuracy",
                "eval_loss",
                "eval_accuracy",
                "elapsed",
            ]
        )

    def load_test_data(self):
        """Carga los datos de prueba en un DataLoader."""

        # datos de pruebas
        train_dataset = TinyImageNetLazy(split="valid")
        sampler = ShardSampler(
            dataset_size=len(train_dataset),
            rank=0,
            world_size=1,
            batch_size=self.batch_size,
        )

        # Obtener todos los índices del shard de una vez, cargar todo los datos
        shard = sampler.get_shard_indices(0)
        n = (len(shard) // self.batch_size) * self.batch_size
        shard = shard[:n]

        # Un solo loader para cada época
        subset = Subset(train_dataset, shard.tolist())
        self.loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=False,
        )

    def results(self):
        """Guarda las métricas en un archivo Excel y genera gráfico de resultados."""
        save_path = self.save_path

        if save_path is None:
            plot_grid(
                history=[
                    (
                        # unir train/test en una sola gráfica
                        (self.metrics["loss"][i], self.metrics["eval_loss"][i]),
                        (
                            (
                                self.metrics["accuracy"][i],
                                self.metrics["eval_accuracy"][i],
                            )
                        ),
                    )
                    for i in range(len(self.metrics))
                ],
                labels=[
                    ("Loss", "Train", "Test"),
                    ("Accuracy", "Train", "Test"),
                ],
                n_cols=1,
                save_path=save_path,
            )

            return

        os.makedirs(save_path, exist_ok=True)

        self.metrics.to_excel(os.path.join(save_path, "metrics_server.xlsx"))

        description = self.metrics.describe(percentiles=[0.1, 0.5, 0.9])
        description.to_excel(
            os.path.join(save_path, "description_server.xlsx"),
            index=True,
        )

        with open(os.path.join(save_path, "train_params.txt"), "w") as f:
            f.write(f"epochs: {self.epochs}\n")
            f.write(f"lr: {self.lr}\n")
            f.write(f"min_workers: {self.min_workers}\n")
            f.write(f"batch_size: {self.batch_size}\n")
            f.write(f"Final accuracy: {self.metrics['eval_accuracy'].iloc[-1]}")

        _, conf = self.evaluate_classification()
        per_class_acc = conf.diag() / conf.sum(dim=1).clamp(min=1)
        excel_report(per_class_acc, conf, self.loader, save_path)

    def evaluate(self) -> tuple[float, float]:
        """Evalua el modelo en el test set y devuelve loss y accuracy."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():  # no se actualizan los pesos
            for x, y in self.loader:
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
        confusion_matrix = MulticlassConfusionMatrix(num_classes=200).to(self.device)

        with torch.no_grad():
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)

                total += y.size(0)
                correct += (predicted == y).sum().item()
                confusion_matrix.update(predicted, y)

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
        Promedio ponderado de los pesos de los workers.

        FedAvg:
            w_new = w_global + η * Σ (n_i/N)

        results = [{delta, samples}]
        """
        N_total = sum(r["samples"] for r in results)

        accum_delta = {}

        for i in range(len(results)):
            n_i = results[i]["samples"]
            weight = n_i / N_total
            delta_i = results[i]["delta"]

            for k, d in delta_i.items():
                d = torch.as_tensor(d, device=self.device)

                if k not in accum_delta:
                    accum_delta[k] = weight * d
                else:
                    accum_delta[k] += weight * d

        # Aplicar delta al modelo global
        state = self.model.state_dict()

        for k in state:
            state[k] = state[k] + accum_delta[k]

        self.model.load_state_dict(state)

    def _aggregate_metrics(self, results):
        """
        Agrega las métricas de los resultados de test de los workers.
        """
        N = sum(r["samples"] for r in results)
        loss = sum(r["loss"] * r["samples"] for r in results) / N
        accuracy = sum(r["accuracy"] * r["samples"] for r in results) / N

        # eval distribuido — sumar counts crudos, no promediar promedios
        eval_total = sum(r["eval_total"] for r in results)
        eval_loss = sum(r["eval_loss"] for r in results) / eval_total
        eval_accuracy = sum(r["eval_correct"] for r in results) / eval_total

        return loss, accuracy, eval_loss, eval_accuracy

    def step(self):
        """
        Ejecuta un paso de entrenamiento distribuido.
        Espera a que los workers estén listos, envía los pesos actuales,
        luego envía el mensaje de step y recopila los resultados y promedia los pesos.
        """
        n_workers = self._wait_and_register_workers()

        if n_workers is None:
            log.warning("Timeout esperando workers (saltar época)")
            return

        t0 = time.perf_counter()

        # pesos y parámetros del modelo en pytorch
        state = {
            k: v.detach().cpu().numpy().astype(np.float16)
            for k, v in self.model.state_dict().items()
        }

        self._broadcast_weights(state)
        self._send_assign(n_workers, self.current_epoch)
        self._broadcast_step(self.current_epoch)
        results = self._collect_results()

        if not results:
            log.warning("No se recibieron resultados, saltando época")
            return

        # result {type, payload} obtner solo el payload
        results = [r["payload"] for r in results]
        self._aggregate(results)

        # métricas
        loss, accuracy, eval_loss, eval_accuracy = self._aggregate_metrics(results)
        elapsed = time.perf_counter() - t0

        self.metrics.loc[self.current_epoch] = [
            self.current_workers,
            len(results),
            loss,
            accuracy,
            eval_loss,
            eval_accuracy,
            elapsed,
        ]

        log.info(
            f"Epoch {self.current_epoch + 1}/{self.epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f} - eval_loss: {eval_loss:.4f} - eval_accuracy: {eval_accuracy:.4f} - elapsed: {format_elapse(elapsed)}"
        )

        self.current_epoch += 1

    @time_wrapper
    def train(self):
        """Entrena el modelo durante el número de épocas especificado."""
        while self.current_epoch < self.epochs:
            self.step()

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
            if self.save_path:
                os.makedirs(self.save_path, exist_ok=True)

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

            self.stop_server()
