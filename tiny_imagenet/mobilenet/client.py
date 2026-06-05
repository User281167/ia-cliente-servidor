from async_impl import AsyncGradWorker
from ddp.pickle_utils import log, send_msg
from tiny_imagenet.load_data import TinyImageNetLazy
from tiny_imagenet.mobilenet.model import (
    get_tiny_imagenet_mobilenet,
    mobilenet_transform,
)


class MobileNetWorker(AsyncGradWorker):
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
                get_tiny_imagenet_mobilenet(lr=lr, device=self.device)
            )

            # Agregar en el worker justo después de recibir config
            print(f"Device worker: {self.device}")
            print(f"Model device: {next(self.model.parameters()).device}")

            self.dataset = TinyImageNetLazy(
                split="train", transform=mobilenet_transform()
            )
            self.test_dataset = TinyImageNetLazy(
                split="valid", transform=mobilenet_transform()
            )

            self.load_samplers(preload=True)
            self.stop = False
            send_msg(self._sock, {"type": "ready", "worker_id": self._worker_id})
