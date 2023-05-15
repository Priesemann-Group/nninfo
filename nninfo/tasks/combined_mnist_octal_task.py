import torch
import torchvision.datasets

from .task import Task, octal_encode_label

class CombinedMnistOctalTask(Task):
    task_id = "combined_mnist_octal_dat"

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
        return 2

    def load_samples(self):
        mnist = torchvision.datasets.MNIST(
            root="../", download=True, train=True)
        mnist_test = torchvision.datasets.MNIST(
            root="../", download=True, train=False)

        fmnist = torchvision.datasets.FashionMNIST(
            root="../", download=True, train=True)
        fmnist_test = torchvision.datasets.FashionMNIST(
            root="../", download=True, train=False)

        x = torch.cat([mnist.data, fmnist.data, mnist_test.data,
                       fmnist_test.data]).reshape(-1, 784) / 256.
        y = torch.cat([mnist.targets, fmnist.targets + 10, mnist_test.targets,
                       fmnist_test.targets + 10])

        y_binary = octal_encode_label(y, 2)

        return x.type(torch.float32), y_binary.type(torch.float32)