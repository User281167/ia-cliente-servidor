import os

import torch

from async_sgd.sgd import AsyncWeightsServer
from cifar10.load_data import cifar10_classes, cifar10_data_len, get_cifar10_dataloader
from utils import plot_confusion_matrix

from .model import get_cifar10_resnet18_model


class CIFAR10Server(AsyncWeightsServer):
    """
    Servidor CIFAR-10 con ResNet18 usando async delta de pesos.
    """

    def __init__(
        self,
        epochs: int = 20,
        lr: float = 0.01,
        gamma: float = 0.5,
        shard_size: int = 2048,
        batch_size: int = 128,
        max_staleness: int = 10,
        test_each: int = 10,
        min_workers: int = 1,
        weight_decay: float = 5e-4,
        save_path: str | None = None,
    ):
        config = {
            "gray": False,
            "normalize": True,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
        }

        super().__init__(
            data_len=cifar10_data_len(),
            epochs=epochs,
            lr=lr,
            gamma=gamma,
            shard_size=shard_size,
            batch_size=batch_size,
            max_staleness=max_staleness,
            test_each=test_each,
            min_workers=min_workers,
            config=config,
            save_path=save_path,
        )

        self.gray = False
        self.normalize = True
        self.weight_decay = weight_decay

        self.model, self.criterion, self.optimizer, _ = get_cifar10_resnet18_model(
            lr=lr, num_classes=len(cifar10_classes), device=self.device
        )

        self.test_loader = get_cifar10_dataloader(
            train=False,
            gray=False,
            normalize=True,
            batch_size=batch_size,
        )

    def _apply_delta(self, delta: dict, gamma: float) -> float:
        state = self.model.state_dict()
        delta_norm_sq = 0.0

        for name, value in delta.items():
            if name not in state:
                continue

            d_t = torch.as_tensor(
                value,
                dtype=state[name].dtype,
                device=state[name].device,
            )

            if (
                self.weight_decay > 0
                and "weight" in name
                and "bn" not in name
                and "bias" not in name
            ):
                state[name] = state[name] * (1 - self.weight_decay) + gamma * d_t
            else:
                state[name] = state[name] + gamma * d_t

            delta_norm_sq += torch.linalg.vector_norm(d_t.float()).item() ** 2

        self.model.load_state_dict(state)
        return delta_norm_sq**0.5

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
                f.write(f"weight_decay: {self.weight_decay}\n")
                f.write(f"final_accuracy: {acc}\n")
