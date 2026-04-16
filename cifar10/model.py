import torch.nn as nn


class Cifar10Model(nn.Module):
    def __init__(self, gray=True, conv=False):
        super().__init__()

        if conv:
            self.net = nn.Sequential(
                nn.Conv2d(1 if gray else 3, 32, 3, padding=1),
                nn.LeakyReLU(),
                nn.Dropout2d(p=0.2),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 128),
                nn.Dropout(p=0.2),
                nn.LeakyReLU(),
                nn.Linear(128, 10),
            )
        else:
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1 * 32 * 32 if gray else 3 * 32 * 32, 128),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 10),
            )

    def forward(self, x):
        return self.net(x)
