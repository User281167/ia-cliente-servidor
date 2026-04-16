import numpy as np
import pandas as pd
import torch

from ddp import DDPServer
from ddp.logger import log
from ddp.message import DDPMessage
from ddp.pickle_utils import send_msg

from .load_data import get_cifar10_dataloader
from .model import Cifar10Model


class CIFAR10Server(DDPServer):
    def __init__(
        self,
        gray: bool = False,
        normalize: bool = False,
        conv: bool = False,
        epochs: int = 20,
        lr: float = 0.001,
        workers: int = 1,
        min_workers: int = 1,
    ):
        super().__init__(workers, min_workers)
        self.model = Cifar10Model(gray=gray, conv=conv)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.current_epoch = 0

        self.test_loader = get_cifar10_dataloader(
            train=False, gray=gray, normalize=normalize
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.metrics = pd.DataFrame(
            columns=["loss", "accuracy", "eval_loss", "eval_accuracy"]
        )

    def evaluate(self):
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return total_loss / total, correct / total

    def _send_assign(self, n_workers: int, epoch: int):
        with self._workers_lock:
            items = list(self._workers.items())

        for i, (wid, sock) in enumerate(items):
            msg = DDPMessage.assign(
                worker_id=wid,
                rank=i,
                world_size=n_workers,
                epoch=epoch,
            )

            self._assignments[wid] = msg

            try:
                send_msg(sock, msg)
            except Exception as e:
                log.warning(f"Worker {wid} fallo assign: {e}")

    def step(self):
        n_workers = self._wait_workers()

        if n_workers is None:
            log.warning("Timeout esperando workers (saltar época)")
            return

        state = {
            k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in self.model.state_dict().items()
        }

        self._broadcast_weights(state)
        self._send_assign(n_workers, self.current_epoch)
        self._broadcast_step(self.current_epoch)
        results = self._collect_results()

        if not results:
            return

        accum_grads: dict[str, torch.Tensor] = {}

        for msg in results:
            grads = msg["payload"]["grads"]

            for k, g in grads.items():
                g = torch.as_tensor(g)

                if k not in accum_grads:
                    accum_grads[k] = g.clone()
                else:
                    accum_grads[k] += g

        # promedio
        for k in accum_grads:
            accum_grads[k] /= len(results)

        # aplicar gradientes
        for name, param in self.model.named_parameters():
            param.grad = accum_grads[name].detach().clone()

        # optimizar
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = float(sum(r["payload"]["loss"] for r in results) / len(results))
        accuracy = float(sum(r["payload"]["accuracy"] for r in results) / len(results))

        eval_loss, eval_accuracy = self.evaluate()

        self.metrics.loc[self.current_epoch] = [
            loss,
            accuracy,
            eval_loss,
            eval_accuracy,
        ]

        log.info(
            f"Epoch {self.current_epoch} - loss: {loss:.4f} - accuracy: {accuracy:.4f} - eval_loss: {eval_loss:.4f} - eval_accuracy: {eval_accuracy:.4f}"
        )

        self.current_epoch += 1

    def train(self):
        while self.current_epoch < self.epochs:
            self.step()

    def run(self, host: str = "0.0.0.0", port: int = 9999):
        self.start_server(host=host, port=port)

        try:
            self.train()
        finally:
            self.stop_server()
