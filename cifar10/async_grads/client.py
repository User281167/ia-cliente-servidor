from async_impl import AsyncGradWorker
from cifar10.load_data import preload_cifar10_to_ram
from cifar10.model import cifar10_get_model
from ddp.pickle_utils import log, send_msg


class CIFAR10Worker(AsyncGradWorker):
    """
    Worker CIFAR-10 para ASGD por gradientes.
    Solo define modelo y datasets; loop async vive en AsyncGradWorker.
    """

    def _register_handlers(self):
        super()._register_handlers()

        @self.on("config")
        def on_config(msg):
            log.info(f"Recibido mensaje de configuracion: {msg}")

            payload = msg["payload"]
            gray = payload["gray"]
            normalize = payload["normalize"]
            conv = payload["conv"]
            lr = payload["lr"]
            self.batch_size = payload["batch_size"]

            self.model, self.criterion, self.optimizer = cifar10_get_model(
                gray=gray,
                conv=conv,
                lr=lr,
                device=self.device,
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
