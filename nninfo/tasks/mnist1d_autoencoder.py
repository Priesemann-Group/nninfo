import torch
import torchvision.datasets

from .task import Task

class Mnist1DTaskAutoencoder(Task):
    task_id = "mnist_1d_autoencoder"

    @property
    def finite(self):
        return True

    @property
    def x_limits(self):
        return (0, 1)

    @property
    def y_limits(self):
        return (0, 1)

    @property
    def x_dim(self):
        return 784

    @property
    def y_dim(self):
        return 784

    def load_samples(self):
        mnist = torchvision.datasets.MNIST(
            root="../", download=True, train=True)
        mnist_test = torchvision.datasets.MNIST(
            root="../", download=True, train=False)

        x = torch.cat([mnist.data, mnist_test.data]).reshape(-1, 784) / 256.

        return x.type(torch.float32), x.type(torch.float32)