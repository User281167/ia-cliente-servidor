import os

from cifar10.load_data import cifar10_classes, get_cifar10_dataloader
from cifar10.model import cifar10_get_model
from sinc import SyncGradServer
from utils import plot_confusion_matrix


class CIFAR10Server(SyncGradServer):
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
        save_path: str | None = None,
    ):
        config = {
            "gray": gray,
            "normalize": normalize,
            "conv": conv,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
        }

        super().__init__(
            epochs,
            lr,
            batch_size,
            min_workers,
            config,
            save_path,
            load_model=False,
        )

        self.gray = gray
        self.normalize = normalize
        self.conv = conv
        self.lr = lr
        self.batch_size = batch_size

        self.model, self.criterion, self.optimizer = cifar10_get_model(
            gray=gray, conv=conv, lr=lr, device=self.device
        )

        self.test_loader = get_cifar10_dataloader(
            train=False, gray=gray, normalize=normalize
        )

    def results(self):
        super().results()

        _, conf = self.evaluate_classification()
        plot_confusion_matrix(
            conf,
            save_path=self.save_path,
            class_names=cifar10_classes,
        )

        # argumentos
        if self.save_path:
            with open(os.path.join(self.save_path, "train_params.txt"), "a") as f:
                f.write(f"\ngray: {self.gray}\n")
                f.write(f"normalize: {self.normalize}\n")
                f.write(f"conv: {self.conv}\n")
