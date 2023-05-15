import torch
import numpy as np

from .task import Task

class FakeTask(Task):
    task_id = "fake_dat"

    @property
    def finite(self):
        return True

    @property
    def x_limits(self):
        return "binary"

    @property
    def y_limits(self):
        return "binary"

    @property
    def x_dim(self):
        if "x_dim" in self._kwargs:
            x_dim = self._kwargs["x_dim"]
        else:
            x_dim = 12
        return x_dim

    @property
    def y_dim(self):
        return 1

    def load_samples(self):
        n_bits = self.x_dim
        x = _create_all_possible_n_bit_configurations(n_bits)

        # effectively setting y with x_0=0 to 1
        y = np.zeros(x.shape[0], dtype=np.int)
        y[int(x.shape[0] / 2):] = y[int(x.shape[0] / 2):] + 1
        y = y[:, np.newaxis]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    
def _create_all_possible_n_bit_configurations(n_bits):
    n_samples = 2 ** n_bits
    # create all integer values
    x_int = np.linspace(0, n_samples - 1, n_samples, endpoint=True, dtype=np.uint32)[
        :, np.newaxis
    ]
    # unpack integer values into bits
    x_bit = np.unpackbits(np.flip(x_int.view("uint8")), axis=1)
    # cut bits to x_dim dimensions
    return x_bit[:, x_bit.shape[1] - n_bits:]