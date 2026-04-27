import numpy as np
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


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


class ShardSampler:
    """
    Calcula qué índices le corresponden a un worker para una época dada.
    Shuffle global por seed=epoch -> sin solapamiento entre workers.
    Compatible con world_size variable (cada época puede tener distinto número de workers).
    """

    def __init__(self, dataset_size: int, rank: int, world_size: int, batch_size: int):
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size

    def get_shard_indices(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(seed=epoch)
        indices = rng.permutation(self.dataset_size)
        shard = indices[self.rank :: self.world_size]

        # truncar para evitar batch incompleto
        n = (len(shard) // self.batch_size) * self.batch_size
        return shard[:n]

    def iter_batches(self, epoch: int):
        """Genera listas de índices por minibatch."""
        shard = self.get_shard_indices(epoch)

        for i in range(0, len(shard), self.batch_size):
            yield shard[i : i + self.batch_size]

    def get_loader(self, epoch: int, dataset: Dataset):
        # Obtener todos los índices del shard de una vez
        shard = self.get_shard_indices(epoch)
        n = (len(shard) // self.batch_size) * self.batch_size
        shard = shard[:n]

        # Un solo loader para toda la época
        subset = Subset(dataset, shard.tolist())
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            num_workers=2,  # workers para cargar datos en paralelo
            persistent_workers=True,
            prefetch_factor=2,  # prefetch factor para cargar datos en paralelo
            pin_memory=False,  # NO cargar en memoria, evitar copias innecesarias
        )
