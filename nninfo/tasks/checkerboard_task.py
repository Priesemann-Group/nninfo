import torch
import numpy as np

from .task import Task

class CheckerboardTask(Task):
    task_id = "checkerboard"

    @property
    def finite(self):
        return False

    @property
    def x_limits(self):
        return (0, 1)

    @property
    def y_limits(self):
        return "binary"

    @property
    def x_dim(self):
        return 2

    @property
    def y_dim(self):
        return 1

    def generate_sample(self, rng, condition=None):
        size = self._kwargs['size']
        x = rng.random(2, dtype=np.float32)
        y = (int(x[0] * size[0]) + int(x[1] * size[1])) % 2
        return x, torch.tensor([y], dtype=torch.float)