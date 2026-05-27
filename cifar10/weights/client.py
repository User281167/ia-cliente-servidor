import pandas as pd

from cifar10.load_data import preload_cifar10_to_ram
from cifar10.model import cifar10_get_model
from ddp.pickle_utils import log
from sinc import SincWeightsWorker


class CIFAR10Worker(SincWeightsWorker):
    """
    Cliente worker para el entrenamiento distribuido de CIFAR-10.
    """

    def __init__(self, host, port, save_path):
        super().__init__(host, port, save_path)

        self.dataset = None
        self.test_dataset = None

        self.rank = 0
        self.world_size = 1
        self.batch_size = 128

        self.metrics = pd.DataFrame(
            columns=["loss", "accuracy", "elapse", "throughput"]
        )

        self._register_handlers()

    def _register_handlers(self):
        """
        Registra los manejadores de mensajes del servidor.
        Ejecuta las funciones correspondientes cuando se reciben mensajes del servidor.
        """

        super()._register_handlers()

        @self.on("config")
        def on_config(msg):
            log.info(f"Recibido mensaje de configuración: {msg}")

            payload = msg["payload"]
            gray = payload["gray"]
            normalize = payload["normalize"]
            conv = payload["conv"]
            lr = payload["lr"]
            self.batch_size = payload["batch_size"]

            self.model, self.criterion, self.optimizer = cifar10_get_model(
                gray=gray, conv=conv, lr=lr, device=self.device
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
