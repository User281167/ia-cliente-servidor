import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class TinyImageNetLazy(Dataset):
    """
    Dataset lazy — HuggingFace carga bajo demanda con arrow memory-map.
    No precarga nada en RAM. Solo accede al índice pedido.

    Normalización recomendada
    """

    def __init__(self, split="train"):
        # HuggingFace usa Arrow files (memory-mapped), no carga en RAM
        self.hf_dataset = load_dataset(
            "Maysee/tiny-imagenet",
            split=split,
            keep_in_memory=False,
        )

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]  # lee solo este registro del disco
        image = sample["image"]
        label = sample["label"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.transform(image), label

    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
        )
