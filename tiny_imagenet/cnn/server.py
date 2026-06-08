import os

from sync import SyncWeightsServer
from tiny_imagenet.load_data import TinyImageNetLazy
from tiny_imagenet.utils.report import (
    compute_confusion_matrix_and_accuracy,
    excel_report,
)
from utils import plot_grid

from .model import get_tiny_imagenet_model


class TinyImageNetServer(SyncWeightsServer):
    """
    Servidor Tiny ImageNet CNN con sync weights.
    Test distribuido ocurre en workers; results no calcula matriz de confusion.
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
            "top5": True,
        }

        super().__init__(
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            min_workers=min_workers,
            config=config,
            save_path=save_path,
            load_model=False,
        )

        self.num_classes = 200

        self.WORKER_TIMEOUT = worker_timeout
        self.model, self.criterion, self.optimizer, self.scheduler = (
            get_tiny_imagenet_model(
                lr=lr,
                epochs=epochs,
                device=self.device,
            )
        )

        if save_path and os.path.exists(os.path.join(save_path, "model.pth")):
            self.model.load(os.path.join(save_path, "model.pth"), device=self.device)

        self.test_loader = TinyImageNetLazy(split="valid").get_loader(
            batch_size=batch_size, shuffle=False
        )

    def results(self):
        """Guarda las métricas en un archivo Excel y genera gráfico de resultados."""
        save_path = self.save_path

        if save_path:
            os.makedirs(save_path, exist_ok=True)

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
                    self.metrics["delta_norm"][i],
                )
                for i in range(len(self.metrics))
            ],
            labels=[
                ("Loss", "Train", "Test"),
                ("Accuracy", "Train", "Test"),
                "Weights Norm",
            ],
            n_cols=1,
            save_path=save_path,
        )

        if save_path:
            # evaluate classification
            acc, conf = self.evaluate_classification()

            with open(os.path.join(save_path, "train_params.txt"), "w") as f:
                f.write(f"epochs: {self.epochs}\n")
                f.write(f"lr: {self.lr}\n")
                f.write(f"min_workers: {self.min_workers}\n")
                f.write(f"batch_size: {self.batch_size}\n")
                f.write(f"Final accuracy: {acc}")

            acc, conf, per_class_acc, per_class_top5_acc = (
                compute_confusion_matrix_and_accuracy(
                    self.model, self.test_loader, num_classes=200, device=self.device
                )
            )

            excel_report(
                per_class_acc,
                conf,
                self.test_loader,
                save_path,
                per_class_top5_acc=per_class_top5_acc,
            )
