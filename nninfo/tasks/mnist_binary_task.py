import torch
import torchvision.datasets

from .task import Task

class MnistBinaryTask(Task):
    task_id = "mnist_binary_dat"

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
        return 4

    def load_samples(self):
        mnist = torchvision.datasets.MNIST(
            root="../", download=True, train=True)
        mnist_test = torchvision.datasets.MNIST(
            root="../", download=True, train=False)
        qmnist_test = torchvision.datasets.QMNIST(
            root="../", what='test50k', download=True, train=False)

        x = torch.cat([mnist.data, mnist_test.data,
                       qmnist_test.data]).reshape(-1, 784) / 256.
        y = torch.cat([mnist.targets, mnist_test.targets,
                       qmnist_test.targets[:, 0]])

        y_binary = self.binary(y, 4)

        return x.type(torch.float32), y_binary.type(torch.float32)

    def binary(self, x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()