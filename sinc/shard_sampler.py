import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from ddp import ShardAssignment


class DistributedEpochSampler(Sampler):
    def __init__(self, dataset_size, batch_size):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.indices = []

    def set_epoch(self, seed, rank, world_size):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.dataset_size)

        shard = indices[rank::world_size]
        n = (len(shard) // self.batch_size) * self.batch_size

        self.indices = shard[:n].tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ShardSampler:
    """
    Calcula qué índices le corresponden a un worker para una época dada.
    Shuffle global por seed aleatorio del servidor -> sin solapamiento entre workers.
    Compatible con world_size variable (cada época puede tener distinto número de workers).
    """

    def __init__(self, dataset_size: int, rank: int, world_size: int, batch_size: int):
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size

    def get_shard_indices(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        indices = rng.permutation(self.dataset_size)
        shard = indices[self.rank :: self.world_size]

        # truncar para evitar batch incompleto
        n = (len(shard) // self.batch_size) * self.batch_size
        return shard[:n]

    def iter_batches(self, seed: int):
        """Genera listas de índices por minibatch."""
        shard = self.get_shard_indices(seed)

        for i in range(0, len(shard), self.batch_size):
            yield shard[i : i + self.batch_size]

    def get_loader(self, seed: int, dataset: Dataset):
        # Obtener todos los índices del shard de una vez
        shard = self.get_shard_indices(seed)

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


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class AsyncShardSampler:
    def __init__(self):
        self._epoch = None
        self._seed = None
        self._indices = None

    def _ensure_shuffle(self, dataset, assignment: ShardAssignment):
        """Realizar shuffle de la época actual"""
        if self._epoch != assignment.epoch or self._seed != assignment.seed:
            rng = np.random.default_rng(assignment.seed)

            self._indices = rng.permutation(len(dataset))

            self._epoch = assignment.epoch
            self._seed = assignment.seed

    def get_loader(self, dataset, assignment, test=False):
        if test:
            self._ensure_shuffle(dataset, assignment)

            end = min(
                assignment.start + assignment.length,
                len(dataset),
            )

            shard = self._indices[assignment.start : end]
        else:
            shard = np.random.choice(
                list(range(len(dataset))), assignment.length, replace=False
            )

        subset = IndexedDataset(dataset, shard)

        return DataLoader(
            subset,
            batch_size=assignment.batch_size,
            num_workers=2,
            persistent_workers=False,
            pin_memory=True,
        )
