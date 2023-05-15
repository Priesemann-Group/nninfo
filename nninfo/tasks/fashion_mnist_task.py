import torch
import torchvision.datasets

from .task import Task

class FashionMnistTask(Task):
    task_id = "fashion_mnist_1d_dat"

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
        mnist = torchvision.datasets.FashionMNIST(
            root="../", download=True, train=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root="../", download=True, train=False)

        x = torch.cat([mnist.data, mnist_test.data]).reshape(-1, 784) / 256.
        y = torch.cat([mnist.targets, mnist_test.targets])

        return x.type(torch.float32), y.type(torch.long)