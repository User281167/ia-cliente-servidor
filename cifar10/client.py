import numpy as np
import torch

from ddp import DDPClient
from ddp.pickle_utils import send_msg

from .load_data import preload_cifar10_to_ram
from .model import Cifar10Model


class CIFAR10Worker(DDPClient):
    def __init__(self, host, port, gray=True, normalize=True, lr=0.01):
        super().__init__(host, port)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Cifar10Model(gray=gray).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.dataset = preload_cifar10_to_ram(
            train=True,
            gray=gray,
            normalize=normalize,
        )

        self.rank = 0
        self.world_size = 1

        self._register_handlers()

    def get_batch(self, epoch):
        N = len(self.dataset)

        rng = np.random.default_rng(seed=epoch)

        # shuffle global
        # shard del worker
        indices = rng.permutation(N)
        shard = indices[self.rank :: self.world_size]

        return shard

    def _register_handlers(self):
        @self.on("assign")
        def on_assign(msg):
            payload = msg["payload"]

            self.rank = payload["rank"]
            self.world_size = payload["world_size"]

        @self.on("weights")
        def on_weights(msg):
            state = msg["payload"]

            state_dict = self.model.state_dict()

            for k in state_dict:
                state_dict[k] = torch.tensor(state[k])

            self.model.load_state_dict(state_dict)

        @self.on("step")
        def on_step(msg):
            epoch = msg["epoch"]

            batch_idx = self.get_batch(epoch)
            X, y = self.dataset[batch_idx]

            X = X.to(self.device)
            y = y.to(self.device)

            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model(X)
            loss = self.criterion(logits, y)

            loss.backward()

            grads = {
                name: param.grad.detach().cpu().numpy().astype(np.float32)
                for name, param in self.model.named_parameters()
                if param.grad is not None
            }

            acc = float((logits.argmax(1) == y).float().mean().item())
            loss = float(loss.item())

            print(f"Worker {self.rank}: epoch={epoch}, acc={acc:.4f}, loss={loss:.4f}")

            send_msg(
                self._sock,
                {
                    "type": "result",
                    "payload": {
                        "grads": grads,
                        "loss": loss,
                        "accuracy": acc,
                    },
                },
            )
