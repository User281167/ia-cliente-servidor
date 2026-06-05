from ddp.pickle_utils import log
from sync import SyncWeightsWorker
from tiny_imagenet.load_data import TinyImageNetLazy

from .model import get_tiny_imagenet_model


class TinyImageClient(SyncWeightsWorker):
    """
    Worker Tiny ImageNet CNN con sync weights.
    Usa TinyImageNetLazy y samplers sync del worker base.
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
            self.compute_top5 = payload.get("top5", False)

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
