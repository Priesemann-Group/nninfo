import time

import PIL
import torch
import torchvision.datasets
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path

from .task import Task

class Face2DTaskAugmented(Task):
    task_id = "face_2d_autoencoder_augmented"

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

        current_dir = Path(__file__).parent
        data_dir = current_dir / ".." / ".." / "data" / "faces"

        # Check if cached version exists
        try:
            x = torch.load(data_dir / "face_2d_autoencoder_augmented.pt", weights_only=True)
            if torch.cuda.is_available():
                x = x.cuda()
            return x, x
        except FileNotFoundError:
            pass


        start = time.perf_counter()

        # Rotate the dataset
        x = []
        for flip in [False, True]:
            for angle in [-10, -5, 0, 5, 10]:
                print(f"Loading dataset with {flip=} and {angle=}")
                dataset = torchvision.datasets.ImageFolder(root=data_dir / "lfw-deepfunneled", transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1 if flip else 0),
                    transforms.Lambda(lambda x: transforms.functional.rotate(x, angle, interpolation=InterpolationMode.BILINEAR)),
                    transforms.Resize((40, 40)),
                    transforms.CenterCrop((32, 32)),
                    transforms.ToTensor(),
                ]))
                
                loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=16)
                x.append(next(iter(loader))[0])

        # Concatenate all the datasets
        x = torch.cat(x)
        print(len(x))

        print("Time to load samples: ", time.perf_counter() - start)

        # Shuffle deterministically
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(x))
        x = x[indices]

        # Save for quick loading
        torch.save(x, data_dir / "face_2d_autoencoder_augmented.pt")

        # Move x to gpu if available
        if torch.cuda.is_available():
            x = x.cuda()

        return x, x
    
class Face2DTaskAugmentedLarge(Task):
    task_id = "face_2d_autoencoder_augmented_large"

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
        return 128 * 128 * 3

    @property
    def y_dim(self):
        return 128 * 128 * 3

    def load_samples(self):

        # Check if cached version exists
        try:
            x = torch.load("/mnt/lustre-emmy-ssd/usr/u11172/face_2d_autoencoder_augmented_large.pt")
            if torch.cuda.is_available():
                x = x.cuda()
            return x, x
        except FileNotFoundError:
            pass


        start = time.perf_counter()

        # Rotate the dataset
        x = []
        for flip in [False, True]:
            for angle in [-10, -5, 0, 5, 10]:
                print(f"Loading dataset with {flip=} and {angle=}")
                dataset = torchvision.datasets.ImageFolder(root='/user/ehrlich5/u11172/nninfo/data/faces/lfw-deepfunneled', transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1 if flip else 0),
                    transforms.Lambda(lambda x: transforms.functional.rotate(x, angle, interpolation=InterpolationMode.BILINEAR)),
                    transforms.Resize((150, 150)),
                    transforms.CenterCrop((128, 128)),
                    transforms.ToTensor(),
                ]))
                
                loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=16)
                x.append(next(iter(loader))[0])

        # Concatenate all the datasets
        x = torch.cat(x)
        print(len(x))

        print("Time to load samples: ", time.perf_counter() - start)

        # Shuffle deterministically
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(x))
        x = x[indices]

        # Save for quick loading
        torch.save(x, "/mnt/lustre-emmy-ssd/usr/u11172/face_2d_autoencoder_augmented_large.pt")

        # Move x to gpu if available
        if torch.cuda.is_available():
            x = x.cuda()

        return x, x

if __name__ == "__main__":
    task = Face2DTaskAugmentedLarge()
    samples = task.load_samples()
    print(samples)
    print(samples[0].shape)
    print(samples[0].min(), samples[0].max())
