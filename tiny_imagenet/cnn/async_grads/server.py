import os

from async_sgd.sgd import AsyncGradServer
from tiny_imagenet.utils.report import (
    compute_confusion_matrix_and_accuracy,
    excel_report,
)

from .model import get_tiny_imagenet_model


class TinyImageNetServer(AsyncGradServer):
    """
    Servidor Tiny ImageNet CNN para ASGD por gradientes.
    Test dataset corre en workers; servidor no calcula matriz de confusion.
    """

    def __init__(
        self,
        epochs: int = 20,
        lr: float = 0.001,
        shard_size: int = 512,
        batch_size: int = 128,
        max_staleness: int = 10,
        test_each: int = 10,
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
            data_len=100_000,
            epochs=epochs,
            lr=lr,
            shard_size=shard_size,
            batch_size=batch_size,
            max_staleness=max_staleness,
            test_each=test_each,
            min_workers=min_workers,
            config=config,
            save_path=save_path,
        )

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

    def results(self) -> None:
        super().results()

        if self.save_path:
            path = os.path.join(self.save_path, "model.pth")
            self.model.save(path)

            acc, conf, per_class_acc, per_class_top5_acc = (
                compute_confusion_matrix_and_accuracy(
                    self.model, self.test_loader, num_classes=200, device=self.device
                )
            )

            excel_report(
                per_class_acc,
                conf,
                self.test_loader,
                self.save_path,
                per_class_top5_acc=per_class_top5_acc,
            )
