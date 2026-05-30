import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary


class BasicBlock(nn.Module):
    # BLOQUE RESIDUAL BÁSICO
    # ResNet-18/34
    # Usa dos convoluciones 3x3.

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Rama principal: aprende F(x)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Rama de identidad (skip connection)
        # Si las dimensiones cambian (stride>1 o canales distintos),
        # necesitamos proyectar x al mismo espacio con Conv 1x1.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)  # autopista de gradiente
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # suma residual: F(x) + x

        return self.relu(out)


class ResNet18(nn.Module):
    # RED COMPLETA ResNet-18 SIMPLIFICADA

    def __init__(self, num_classes=10, in_size=(3, 32, 32)):
        super().__init__()
        self.in_size = in_size

        # Stem: entrada inicial (ajustado para 32x32, sin MaxPool agresivo)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 4 stages con bloques residuales
        # Cada stage dobla canales y reduce resolución (stride=2 en el 1er bloque)
        self.layer1 = self._make_layer(64, 64, stride=1)  # 32x32
        self.layer2 = self._make_layer(64, 128, stride=2)  # 16x16
        self.layer3 = self._make_layer(128, 256, stride=2)  # 8x8
        self.layer4 = self._make_layer(256, 256, stride=2)  # 4x4

        # Clasificador
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # → (B, 256, 1, 1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, stride):
        # Cada stage tiene 2 bloques; solo el primero puede cambiar stride/canales
        return nn.Sequential(
            BasicBlock(in_ch, out_ch, stride=stride),
            BasicBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def get_cifar10_resnet18_model(lr=0.01, num_classes=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Modelo ResNet18 creado en {device}")

    model = ResNet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = None

    summary(model, input_size=(3, 32, 32))

    return model, criterion, optimizer, scheduler
