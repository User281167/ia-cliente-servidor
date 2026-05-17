import torch
import torch.nn as nn
from torchsummary import summary


class ConvBNAct(nn.Module):
    """
    ConvBNAct es una secuencia de convolución, batch normalization y activación GELU.

    Batch normalization:
        Batch normalization es una técnica de normalización que se aplica a las salidas de una capa de convolución.
        Normaliza las salidas de la capa para que tengan media 0 y varianza 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TinyImageNetModel(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            ConvBNAct(3, 64),  # expansión de filtros
            ConvBNAct(64, 64),  # # profundización de características
            nn.MaxPool2d(2),
            # Block 2
            ConvBNAct(64, 128),
            ConvBNAct(128, 128),
            nn.MaxPool2d(2),
            # Block 3
            ConvBNAct(128, 256),
            ConvBNAct(256, 256),
            nn.MaxPool2d(2),
            # Block 4
            ConvBNAct(256, 512),
            ConvBNAct(512, 512),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device=None):
        self.load_state_dict(torch.load(path, weights_only=True))

        if device is not None:
            self.to(device)


def get_tiny_imagenet_model(lr=0.001, epochs=20, device=None):
    """
    Returns a Tiny ImageNet model, criterion, and optimizer.

    return:
        model: TinyImageNetModel
        criterion: torch.nn.CrossEntropyLoss
        optimizer: torch.optim.Adam
        scheduler: torch.optim.lr_scheduler
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyImageNetModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    summary(model, input_size=(3, 64, 64))

    return model, criterion, optimizer, scheduler
