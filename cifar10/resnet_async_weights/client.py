from async_impl import AsyncWeightsWorker
from cifar10.load_data import cifar10_classes, preload_cifar10_to_ram
from ddp.pickle_utils import log, send_msg

from .model import get_cifar10_resnet18_model


class CIFAR10Worker(AsyncWeightsWorker):
    """
    Worker CIFAR-10 con ResNet18 usando async delta de pesos.
    """

    def _register_handlers(self):
        super()._register_handlers()

        @self.on("config")
        def on_config(msg):
            log.info(f"Recibido mensaje de configuracion: {msg}")

            payload = msg["payload"]
            gray = payload["gray"]
            normalize = payload["normalize"]
            lr = payload["lr"]
            self.batch_size = payload["batch_size"]

            self.model, self.criterion, self.optimizer, self.scheduler = (
                get_cifar10_resnet18_model(
                    lr=lr, num_classes=len(cifar10_classes), device=self.device
                )
            )

            self.dataset = preload_cifar10_to_ram(
                train=True,
                gray=gray,
                normalize=normalize,
            )
            self.test_dataset = preload_cifar10_to_ram(
                train=False,
                gray=gray,
                normalize=normalize,
            )

            self.load_samplers(preload=True)
            self.stop = False
            send_msg(self._sock, {"type": "ready", "worker_id": self._worker_id})
