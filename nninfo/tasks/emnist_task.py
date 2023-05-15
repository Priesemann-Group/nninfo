import torch
import torchvision.datasets

from .task import Task

class EMnist1DTask(Task):
    task_id = "emnist_1d_dat"

    @property
    def finite(self):
        return True

    @property
    def x_limits(self):
        return (0, 1)

    @property
    def y_limits(self):
        return "binary"

    @property
    def x_dim(self):
        return 784

    @property
    def y_dim(self):
        return 10

    def load_samples(self):
        emnist = torchvision.datasets.EMNIST(
            root=".../", split='digits', download=True, train=True)
        emnist_test = torchvision.datasets.EMNIST(
            root="../", split='digits', download=True, train=False)

        x = torch.cat([torch.transpose(emnist.data, 1, 2), torch.transpose(
            emnist_test.data, 1, 2)]).reshape(-1, 784) / 256
        y = torch.cat([emnist.targets, emnist_test.targets])

        return x.type(torch.float32), y.type(torch.long)