import os
from typing import Literal

from cifar10.load_data import cifar10_classes, cifar10_data_len, get_cifar10_dataloader
from cifar10.model import cifar10_get_model
from rennala_sgd import RennalaSGDServer, RennalaWeightsServer
from utils import plot_confusion_matrix


class CIFAR10ServerBase:
    def __init__(
        self,
        B=10,
        gray=False,
        normalize=False,
        conv=False,
        epochs=20,
        lr=0.001,
        gamma=0.1,
        shard_size=5000,
        batch_size=128,
        test_each=10,
        min_workers=1,
        save_path=None,
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
            B=B,
            epochs=epochs,
            lr=lr,
            gamma=gamma,
            shard_size=shard_size,
            batch_size=batch_size,
            test_each=test_each,
            min_workers=min_workers,
            config=config,
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


class CIFAR10GradServer(CIFAR10ServerBase, RennalaSGDServer):
    pass


class CIFAR10WeightsServer(CIFAR10ServerBase, RennalaWeightsServer):
    pass


server_type = Literal["grad", "weights"]


def get_server(
    server_type: server_type,
    B=10,
    gray=False,
    normalize=False,
    conv=False,
    epochs=20,
    lr=0.001,
    gamma=0.1,
    shard_size=5000,
    batch_size=128,
    test_each=10,
    min_workers=1,
    save_path=None,
) -> CIFAR10GradServer | CIFAR10WeightsServer:
    if server_type == "grad":
        return CIFAR10GradServer(
            B=B,
            gray=gray,
            normalize=normalize,
            conv=conv,
            epochs=epochs,
            lr=lr,
            shard_size=shard_size,
            batch_size=batch_size,
            test_each=test_each,
            min_workers=min_workers,
            save_path=save_path,
        )
    elif server_type == "weights":
        return CIFAR10WeightsServer(
            B=B,
            gray=gray,
            normalize=normalize,
            conv=conv,
            epochs=epochs,
            lr=lr,
            gamma=gamma,
            shard_size=shard_size,
            batch_size=batch_size,
            test_each=test_each,
            min_workers=min_workers,
            save_path=save_path,
        )
    else:
        raise ValueError(f"Invalid server type: {server_type}")
