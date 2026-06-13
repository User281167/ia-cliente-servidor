import os

import torch
from torch.utils.data import DataLoader

from async_sgd.sgd import AsyncWeightsServer
from tiny_imagenet.load_data import TinyImageNetLazy
from tiny_imagenet.resnet.model import (
    get_tiny_imagenet_resnet,
    resnet_transform,
)
from tiny_imagenet.utils.report import (
    compute_confusion_matrix_and_accuracy,
    excel_report,
)


class ResNetServer(AsyncWeightsServer):
    def __init__(
        self,
        epochs=20,
        lr=0.001,
        shard_size=5000,
        batch_size=64,
        max_staleness=10,
        test_each=5,
        min_workers=1,
        save_path=None,
    ):
        config = {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "top5": True,
        }

        self.weight_decay = 5e-4

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

        self.model, self.criterion, self.optimizer, self.scheduler = (
            get_tiny_imagenet_resnet(lr=lr, device=self.device)
        )

        if save_path and os.path.exists(os.path.join(save_path, "model.pth")):
            self.model.load(os.path.join(save_path, "model.pth"), device=self.device)

        test_dataset = TinyImageNetLazy(split="valid", transform=resnet_transform())
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def results(self):
        super().results()

        save_path = self.save_path
        if save_path:
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

            with open(os.path.join(save_path, "train_params.txt"), "a") as f:
                f.write(f"final_accuracy: {acc}\n")

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
