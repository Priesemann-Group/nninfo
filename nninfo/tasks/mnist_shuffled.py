import torch
import torchvision
import numpy as np

from .task import Task

class Mnist1DShuffledTask(Task):
    task_id = "mnist_1d_shuffled_dat"
    
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


    def load_samples(self, component_dir):
        mnist = torchvision.datasets.MNIST(root="../", download=True, train=True)
        mnist_test = torchvision.datasets.MNIST(root="../", download=True, train=False)
        qmnist_test = torchvision.datasets.QMNIST(root="../", what='test50k', download=True, train=False)
        
        x = torch.cat([mnist.data, mnist_test.data, qmnist_test.data]).reshape(-1, 784) / 256.
        y = torch.cat([mnist.targets, mnist_test.targets, qmnist_test.targets[:,0]])

        # Shuffle y labels for test and train set separately
        rng = np.random.default_rng(42 + self._kwargs["seed"])
        y[:60_000] = torch.tensor(rng.permutation(y[:60_000]))
        y[60_000:] = torch.tensor(rng.permutation(y[60_000:]))

        return x.type(torch.float32), y.type(torch.long)