import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

EFFICIENTNET_MEAN = (0.485, 0.456, 0.406)
EFFICIENTNET_STD = (0.229, 0.224, 0.225)


def efficientnet_transform(train=True):
    """
    Se usa resize 224x224 para aprovechar el backbone de EfficientNet.
    """

    if train:
        return transforms.Compose(
            [
                # transforms.Resize(140),
                # transforms.RandomCrop(128),
                # transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(
                #     brightness=0.2, contrast=0.2, saturation=0.2
                # ),
                # transforms.RandomRotation(10),  # rotación leve
                # transforms.RandomGrayscale(p=0.05),  # grayscale ocasional
                transforms.ToTensor(),
                transforms.Normalize(EFFICIENTNET_MEAN, EFFICIENTNET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                # transforms.Resize(140),
                # transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(EFFICIENTNET_MEAN, EFFICIENTNET_STD),
            ]
        )


def get_tiny_imagenet_efficientnet(lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Congelar todas las capas excepto classifier
    for name, param in model.named_parameters():
        if "classifier" in name or "features.8" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Reemplazar la capa final para 200 clases
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 1024),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 512),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, 200),
    )

    model = model.to(device)

    print(f"Modelo cargado en {device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=5e-4)

    total_steps = 20 * 100
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    return model, criterion, optimizer, scheduler
