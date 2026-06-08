import os

from torch.utils.data import DataLoader

from async_impl import AsyncGradServer
from tiny_imagenet.efficientnet.model import (
    efficientnet_transform,
    get_tiny_imagenet_efficientnet,
)
from tiny_imagenet.load_data import TinyImageNetLazy
from tiny_imagenet.utils.report import (
    compute_confusion_matrix_and_accuracy,
    excel_report,
)


class EfficientNetServer(AsyncGradServer):
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
            get_tiny_imagenet_efficientnet(lr=lr, device=self.device)
        )

        if save_path and os.path.exists(os.path.join(save_path, "model.pth")):
            self.model.load(os.path.join(save_path, "model.pth"), device=self.device)

        test_dataset = TinyImageNetLazy(
            split="valid", transform=efficientnet_transform()
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
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
