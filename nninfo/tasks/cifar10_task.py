import torch
import torchvision.datasets

from .task import Task

class CIFAR10Task(Task):
    task_id = "cifar10_1d_dat"

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
        return 32 * 32 * 3

    @property
    def y_dim(self):
        return 10

    def load_samples(self):

        cifar10_train = torchvision.datasets.CIFAR10(
            root="../", download=True, train=True)
        cifar10_test = torchvision.datasets.CIFAR10(
            root="../", download=True, train=False)
        x_train = torch.from_numpy(cifar10_train.data) / 1.
        x_test = torch.from_numpy(cifar10_test.data) / 1.
        x = torch.cat([x_train, x_test])
        x = x - torch.mean(x_train, axis=[0, 1, 2])
        x = x / torch.std(x_train, axis=[0, 1, 2])

        x = torch.transpose(x, 1, 3)

        y = torch.cat([torch.tensor(cifar10_train.targets),
                      torch.tensor(cifar10_test.targets)])

        return x.type(torch.float32), y.type(torch.long)