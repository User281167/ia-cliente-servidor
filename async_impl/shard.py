import numpy as np
from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = []

    def set_indices(self, indices):
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

    def _ensure_shuffle(self, dataset, assignment):
        if self._epoch != assignment.epoch or self._seed != assignment.seed:
            rng = np.random.default_rng(assignment.seed)

            self._indices = rng.permutation(len(dataset))

            self._epoch = assignment.epoch
            self._seed = assignment.seed

    def get_indices(self, dataset, assignment, test=False):
        if test:
            return np.random.choice(
                len(dataset),
                assignment.length,
                replace=False,
            )

        self._ensure_shuffle(dataset, assignment)

        end = min(
            assignment.start + assignment.length,
            len(dataset),
        )

        return self._indices[assignment.start : end]
