import torch
import torchvision.datasets

from .task import Task

class ReducedMnistTask(Task):
    task_id = "mnist_reduced_dat"

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
        mnist = torchvision.datasets.MNIST(
            root="../", download=True, train=True)
        mnist_test = torchvision.datasets.MNIST(
            root="../", download=True, train=False)

        x = torch.cat([mnist.train_data, mnist_test.test_data]
                      ).reshape(-1, 784)
        y = torch.cat([mnist.train_labels, mnist_test.test_labels])

        # Apply filter
        f = (y == 3) | (y == 6) | (y == 8) | (y == 9)
        x = x[f]
        y = y[f]

        return x.type(torch.float32), y.type(torch.long)