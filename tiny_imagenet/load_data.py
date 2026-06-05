from datasets import load_dataset

# problema importanto primero torchvision que datasets con las versiones actuales
print("", end="")
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNetLazy(Dataset):
    def __init__(self, split="train", transform=None):
        hf_dataset = load_dataset(
            "Maysee/tiny-imagenet",
            split=split,
            keep_in_memory=False,
        )

        self.hf_dataset = hf_dataset

        # Precargar todo en listas Python (RAM) — ~500MB train, ~50MB val
        # Esto elimina el I/O de Arrow en cada __getitem__
        print(f"Precargando {len(hf_dataset)} imágenes en RAM...")
        self.images = []
        self.labels = []

        for sample in hf_dataset:
            img = sample["image"]

            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            if img.mode != "RGB":
                img = img.convert("RGB")

            self.images.append(img)
            self.labels.append(sample["label"])

        print("Precarga completa.")

        if transform is not None:
            self.transform = transform
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]
