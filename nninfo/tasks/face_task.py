import torch
import torchvision.datasets
import numpy as np

from torchvision import transforms
from tqdm import tqdm

from .task import Task

class Face2DTask(Task):
    task_id = "face_2d_autoencoder"

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
        return 32 * 32 * 3

    def load_samples(self):
        dataset = torchvision.datasets.ImageFolder(root='/user/ehrlich5/u11172/nninfo/data/faces/lfw-deepfunneled', transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]))
        print(len(dataset))

        loader= torch.utils.data.DataLoader(dataset, batch_size=13233, shuffle=False)
        x = next(iter(loader))[0]
        print(x.shape)

        # Shuffle deterministically
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(x))
        x = x[indices]

        # Move x to gpu if available
        if torch.cuda.is_available():
            x = x.cuda()

        return x, x
    
class Face1DTask:
    task_id = "face_1d_autoencoder"

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
        return 45 * 45 * 3

    @property
    def y_dim(self):
        return 45 * 45 * 3

    def load_samples(self):
        dataset = torchvision.datasets.ImageFolder(root='/user/ehrlich5/u11172/nninfo/data/faces/lfw-deepfunneled', transform=transforms.Compose([
            transforms.Resize((45, 45)),
            transforms.ToTensor(),
        ]))
        print(len(dataset))

        loader= torch.utils.data.DataLoader(dataset, batch_size=13233, shuffle=False)
        x = next(iter(loader))[0]

        # Shuffle deterministically
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(x))
        x = x[indices]

        # Move x to gpu if available
        if torch.cuda.is_available():
            x = x.cuda()

        return x.view(len(x), -1), x.view(len(x), -1)


if __name__ == "__main__":
    task = Face2DTask()
    samples = task.load_samples()
    print(samples)
    print(samples[0].shape)
    print(samples[0].min(), samples[0].max())
