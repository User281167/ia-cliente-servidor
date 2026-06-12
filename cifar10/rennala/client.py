from pandas.core.resample import Literal

from async_impl import AsyncGradWorker, AsyncWeightsWorker
from cifar10.load_data import preload_cifar10_to_ram
from cifar10.model import cifar10_get_model
from ddp.pickle_utils import send_msg


class CIFAR10WorkerBase:
    def _register_handlers(self):
        super()._register_handlers()

        @self.on("config")
        def on_config(msg):
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

            send_msg(
                self._sock,
                {"type": "ready", "worker_id": self._worker_id},
            )


class CIFAR10GradWorker(CIFAR10WorkerBase, AsyncGradWorker):
    pass


class CIFAR10WeightsWorker(CIFAR10WorkerBase, AsyncWeightsWorker):
    pass


worker_type = Literal["grad", "weights"]


def get_worker(
    worker_type: worker_type, host, port, save_path
) -> CIFAR10GradWorker | CIFAR10WeightsWorker:
    if worker_type == "grad":
        return CIFAR10GradWorker(host=host, port=port, save_path=save_path)
    elif worker_type == "weights":
        return CIFAR10WeightsWorker(host=host, port=port, save_path=save_path)
    else:
        raise ValueError(f"Unknown worker type: {worker_type}")
