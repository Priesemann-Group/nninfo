import torch
import torchvision.datasets

from .task import Task

class MnistEightBinaryTask(Task):
    """Mnist task but only with digits from 0-7

    len train: 48200
    len test: 48275

    """

    task_id = "mnist8_binary_dat"

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
        return 3

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

        # Filter out digits 8 and 9
        filter_mask = (y != 8) & (y != 9)
        y = y[filter_mask]
        x = x[filter_mask]

        y_binary = self.binary(y, 3)

        return x.type(torch.float32), y_binary.type(torch.float32)

    def binary(self, x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()