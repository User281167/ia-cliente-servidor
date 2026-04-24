import torch
import torch.nn as nn
from torchinfo import summary


class Cifar10Model(nn.Module):
    """
    Modelo para el entrenamiento distribuido de CIFAR-10.
    Red simple convolucional o fully connected.

    Uso de:
        - LeakyReLU evitar vanishing gradient neuronas que mantienen inactivas
        - Dropout evitar overfitting
        - Conv2d convolución 2D para extraer características de las imágenes
    """

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
                nn.Dropout(p=0.4),
                nn.LeakyReLU(),
                nn.Linear(128, 10),
            )
        else:
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1 * 32 * 32 if gray else 3 * 32 * 32, 128),
                nn.LeakyReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(128, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 10),
            )

    def forward(self, x):
        return self.net(x)


def cifar10_get_model(gray=True, conv=False, lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Cifar10Model(gray=gray, conv=conv).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    summary(model, input_size=(1, 1 if gray else 3, 32, 32))

    return model, criterion, optimizer
