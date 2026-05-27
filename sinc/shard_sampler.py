import numpy as np
from torch.utils.data import Sampler


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
