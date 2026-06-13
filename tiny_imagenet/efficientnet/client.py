from async_sgd.sgd import AsyncGradWorker
from ddp.pickle_utils import log, send_msg
from tiny_imagenet.efficientnet.model import (
    efficientnet_transform,
    get_tiny_imagenet_efficientnet,
)
from tiny_imagenet.load_data import TinyImageNetLazy


class EfficientNetWorker(AsyncGradWorker):
    def _register_handlers(self):
        super()._register_handlers()

        @self.on("config")
        def on_config(msg):
            log.info(f"Recibido mensaje de configuracion: {msg}")

            payload = msg["payload"]
            lr = payload["lr"]
            self.batch_size = payload["batch_size"]
            self.compute_top5 = payload.get("top5", False)

            self.model, self.criterion, self.optimizer, self.scheduler = (
                get_tiny_imagenet_efficientnet(lr=lr, device=self.device)
            )

            print(f"Device worker: {self.device}")
            print(f"Model device: {next(self.model.parameters()).device}")

            self.dataset = TinyImageNetLazy(
                split="train", transform=efficientnet_transform()
            )
            self.test_dataset = TinyImageNetLazy(
                split="valid", transform=efficientnet_transform()
            )

            self.load_samplers(preload=True)
            self.stop = False
            send_msg(self._sock, {"type": "ready", "worker_id": self._worker_id})
