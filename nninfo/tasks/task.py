import numpy as np
import torch.utils.data
import torchvision
from abc import ABC, abstractmethod

import nninfo
log = nninfo.logger.get_logger(__name__)

class Task(ABC):

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    _subclasses = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclasses[cls.task_id] = cls

    @staticmethod
    def list_available_tasks():
        return list(Task._subclasses.keys())

    @property
    def kwargs(self):
        return self._kwargs

    @property
    @abstractmethod
    def finite(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def task_id(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def x_limits(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def y_limits(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def x_dim(self):
        """
        Returns the dimension of the feature component of a single sample
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def y_dim(self):
        """
        Returns the dimension of the label component of a single sample
        """
        raise NotImplementedError

    @classmethod
    def from_id(cls, task_id, **kwargs):
        return cls._subclasses[task_id](**kwargs)

    def generate_sample(self, rng, condition=None):
        raise NotImplementedError(
            "Finite-sample tasks do not support the generation of samples."
        )

    @abstractmethod
    def load_samples(self):
        raise NotImplementedError

def binary_encode_label(y, bits):
    """Encode the label tensor y as a tensor of bits."""
    mask = 2**torch.arange(bits).to(y.device, y.dtype)
    return y.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def quaternary_encode_label(y, quits):
    """Encode the label tensor y as a tensor of quits."""
    binary = binary_encode_label(y, 2*quits)
    return binary[:, ::2] + 2 * binary[:, 1::2]


def octal_encode_label(y, octs):
    """Encode the label tensor y as a tensor of octs."""
    quaternary = quaternary_encode_label(y, 2*octs)
    return quaternary[:, ::2] + 4 * quaternary[:, 1::2]