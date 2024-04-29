import numpy as np

from ..Variables import Tensor


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        if isinstance(dataset, Tensor):
            self.dataset = [dataset]
        elif isinstance(dataset, list) and all(isinstance(i, Tensor) for i in dataset):
            if not all(len(i) == len(dataset[0]) for i in dataset):
                raise ValueError("All tensors should have the same batch_size")
            self.dataset = dataset
        else:
            raise ValueError("Dataset should be a Tensor or a list of Tensors")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = list(range(len(self.dataset[0])))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        for i in range(0, len(self.dataset[0]), self.batch_size):
            yield [d[self.indexes[i : i + self.batch_size]] for d in self.dataset]

    def __len__(self):
        return (len(self.dataset[0]) + self.batch_size - 1) // self.batch_size
