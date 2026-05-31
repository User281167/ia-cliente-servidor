import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

MOBILENET_MEAN = (0.485, 0.456, 0.406)
MOBILENET_STD = (0.229, 0.224, 0.225)


def mobilenet_transform(train=True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize(70),  # ligeramente mayor para crop
                transforms.RandomCrop(64),  # crop aleatorio → augmentation
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(MOBILENET_MEAN, MOBILENET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(MOBILENET_MEAN, MOBILENET_STD),
            ]
        )


def get_tiny_imagenet_mobilenet(lr=0.01, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # Congelar backbone completo
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280

    # Clasificador con una capa intermedia — más capacidad que un Linear simple
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, 200),
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainable = filter(lambda p: p.requires_grad, model.parameters())

    # Adam converge más rápido que SGD cuando solo se entrena el clasificador
    optimizer = optim.Adam(trainable, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    return model, criterion, optimizer, scheduler
