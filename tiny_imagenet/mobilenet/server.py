import os

from torch.utils.data import DataLoader

from async_impl import AsyncGradServer
from tiny_imagenet.load_data import TinyImageNetLazy
from tiny_imagenet.mobilenet.model import (
    get_tiny_imagenet_mobilenet,
    mobilenet_transform,
)
from tiny_imagenet.utils.report import excel_report


class MobileNetServer(AsyncGradServer):
    def __init__(
        self,
        epochs=20,
        lr=0.001,
        shard_size=5000,
        batch_size=128,
        max_staleness=10,
        test_each=10,
        min_workers=1,
        save_path=None,
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

        self.model, self.criterion, self.optimizer, self.scheduler = (
            get_tiny_imagenet_mobilenet(lr=lr, device=self.device)
        )

        test_dataset = TinyImageNetLazy(split="valid", transform=mobilenet_transform())
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    def results(self):
        super().results()

        save_path = self.save_path
        if save_path:
            acc, conf = self.evaluate_classification(num_classes=200)
            per_class_acc = conf.diag() / conf.sum(dim=1).clamp(min=1)
            excel_report(per_class_acc, conf, self.test_loader, save_path)

            with open(os.path.join(save_path, "train_params.txt"), "a") as f:
                f.write(f"final_accuracy: {acc}\n")
