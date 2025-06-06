import torch
import torchvision.datasets

from .task import Task

class Mnist2DTask(Task):
    task_id = "mnist_2d_dat"

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
        #qmnist_test = torchvision.datasets.QMNIST(
        #    root="../", what='test50k', download=True, train=False)

        x = torch.cat([mnist.data, mnist_test.data]).reshape(-1, 1, 28, 28) / 256.
        y = torch.cat([mnist.targets, mnist_test.targets])
        
        # move to gpu if available
        if torch.cuda.is_available():
            return x.type(torch.float32).cuda(), y.type(torch.long).cuda() 
         
        return x.type(torch.float32), y.type(torch.long)
    
    def plot_sample(self, x, y):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(x.reshape(28, 28), cmap="gray")
        ax.set_title(f"Label: {y}")