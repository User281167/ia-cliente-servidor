from async_impl import AsyncGradWorker
from ddp.pickle_utils import log, send_msg
from tiny_imagenet.load_data import TinyImageNetLazy

from .model import get_tiny_imagenet_model


class TinyImageNetClient(AsyncGradWorker):
    """
    Worker Tiny ImageNet CNN para ASGD por gradientes.
    Modelo/datasets aqui; loop async vive en AsyncGradWorker.
    """

    def _register_handlers(self):
        super()._register_handlers()

        @self.on("config")
        def on_config(msg):
            log.info(f"Recibido mensaje de configuracion: {msg}")

            payload = msg["payload"]
            lr = payload["lr"]
            epochs = payload["epochs"]
            self.batch_size = payload["batch_size"]

            self.model, self.criterion, self.optimizer, self.scheduler = (
                get_tiny_imagenet_model(
                    lr=lr,
                    epochs=epochs,
                    device=self.device,
                )
            )

            self.dataset = TinyImageNetLazy(split="train")
            self.test_dataset = TinyImageNetLazy(split="valid")
            self.load_samplers(preload=False)

            self.stop = False
            send_msg(self._sock, {"type": "ready", "worker_id": self._worker_id})
