import os

from async_impl import AsyncGradServer
from cifar10.load_data import cifar10_classes, cifar10_data_len, get_cifar10_dataloader
from cifar10.model import cifar10_get_model
from utils import plot_confusion_matrix


class CIFAR10Server(AsyncGradServer):
    """
    Servidor CIFAR-10 para ASGD por gradientes.
    La logica async generica vive en AsyncGradServer.
    """

    def __init__(
        self,
        gray: bool = False,
        normalize: bool = False,
        conv: bool = False,
        epochs: int = 20,
        lr: float = 0.001,
        shard_size: int = 5000,
        batch_size: int = 128,
        max_staleness: int = 10,
        test_each: int = 10,
        min_workers: int = 1,
        save_path: str | None = None,
        use_lr_decay: bool = False,
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
            data_len=cifar10_data_len(),
            epochs=epochs,
            lr=lr,
            shard_size=shard_size,
            batch_size=batch_size,
            max_staleness=max_staleness,
            test_each=test_each,
            min_workers=min_workers,
            config=config,
            use_lr_decay=use_lr_decay,
            save_path=save_path,
        )

        self.gray = gray
        self.normalize = normalize
        self.conv = conv

        self.model, self.criterion, self.optimizer = cifar10_get_model(
            gray=gray,
            conv=conv,
            lr=lr,
            device=self.device,
        )
        self.test_loader = get_cifar10_dataloader(
            train=False,
            gray=gray,
            normalize=normalize,
            batch_size=batch_size,
        )

    def results(self) -> None:
        super().results()

        acc, conf = self.evaluate_classification(num_classes=len(cifar10_classes))
        plot_confusion_matrix(
            conf,
            save_path=self.save_path,
            class_names=cifar10_classes,
        )

        if self.save_path:
            with open(os.path.join(self.save_path, "train_params.txt"), "a") as f:
                f.write(f"gray: {self.gray}\n")
                f.write(f"normalize: {self.normalize}\n")
                f.write(f"conv: {self.conv}\n")
                f.write(f"final_accuracy: {acc}\n")
