import torch
import numpy as np

from .task import Task

class ParityTask(Task):
    task_id = "parity"

    @property
    def finite(self):
        return False

    @property
    def x_limits(self):
        return (0, 1) if self._kwargs["continuous"] else "binary"

    @property
    def y_limits(self):
        return "binary"

    @property
    def x_dim(self):
        return self._kwargs["n_bits"]

    @property
    def y_dim(self):
        return 1

    def generate_sample(self, rng, condition=None):
        n_bits = self._kwargs["n_bits"]

        if self._kwargs["continuous"]:
            x = rng.random(size=n_bits, dtype=np.float32)
            y = (x >= 0.5).sum() % 2
        else:
            x = rng.integers(2, size=n_bits)
            y = x.sum() % 2

        return torch.tensor(x, dtype=torch.float), torch.tensor([y], dtype=torch.float)