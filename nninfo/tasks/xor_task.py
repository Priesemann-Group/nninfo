import torch
import numpy as np

from .task import Task

class XorTask(Task):
    task_id = "xor_dat"

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
        x = rng.random(2, dtype=np.float32)
        y = (x[0] >= 0.5) ^ (x[1] >= 0.5)
        return x, torch.tensor([y], dtype=torch.float)