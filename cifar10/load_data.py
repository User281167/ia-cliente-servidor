import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

cifar10_classes = (
    "avión",
    "automóvil",
    "pájaro",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camión",
)


def preload_cifar10_to_ram(train=True, gray=False, normalize=True):
    """Cargar  CIFAR-10 en RAM como un TensorDataset."""
    transform_list = []

    if gray:
        transform_list.append(transforms.Grayscale())

    transform_list.append(transforms.ToTensor())

    if normalize:
        mean = (0.5,) if gray else (0.5, 0.5, 0.5)
        std = (0.5,) if gray else (0.5, 0.5, 0.5)
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    transform = transforms.Compose(transform_list)

    dataset = datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transform
    )

    # Cargar en RAM
    loader = DataLoader(dataset, batch_size=len(dataset), num_workers=0, shuffle=False)
    images, labels = next(iter(loader))

    return TensorDataset(images, labels)


def get_cifar10_dataloader(train=True, gray=False, batch_size=128, normalize=True):
    tensor_dataset = preload_cifar10_to_ram(train=train, gray=gray, normalize=normalize)
    return DataLoader(
        tensor_dataset, batch_size=batch_size, shuffle=train, num_workers=0
    )


def plot_images(gray=True, size=10):
    dataloader = get_cifar10_dataloader(
        gray=gray, train=True, batch_size=size, normalize=False
    )
    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))

    for i, (image, label) in enumerate(zip(images, labels)):
        if gray:
            axes[i].imshow(image.squeeze(), cmap="gray")
        else:
            axes[i].imshow(image.permute(1, 2, 0))

        axes[i].set_title(cifar10_classes[label.item()])
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
