import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights, resnet18

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)


def resnet_transform(train=True):
    if train:
        return transforms.Compose(
            [
                # transforms.Resize(
                #     256
                # ),  # Redimensionar a 256 para luego hacer RandomCrop a 224
                # transforms.RandomCrop(224),  # ResNet fue preentrenado en 224x224
                # transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(
                #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                # ),
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
            ]
        )


def get_tiny_imagenet_resnet(lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar ResNet50 preentrenado
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Congelar todas las capas excepto las ultimas 2 capas de la red (layer3 y layer4)
    # Esto permite que el modelo adapte mejor las caracteristicas de alto nivel a Tiny ImageNet
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Reemplazar la capa final para 200 clases
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, 200),
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Solo entrenar los parametros que no estan congelados
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    # Usar AdamW que es mejor que Adam para regularizacion
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=5e-4)

    # Scheduler con warmup inicial
    # OneCycleLR usa total_steps
    total_steps = 20 * 100  # 20 epochs * ~100 steps per epoch (estimado)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,  # Aprendizaje maximo
        total_steps=total_steps,
        pct_start=0.1,  # 10% del entrenamiento para warmup
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    return model, criterion, optimizer, scheduler
