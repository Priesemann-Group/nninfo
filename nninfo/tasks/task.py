from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision
from abc import ABC, abstractmethod

import nninfo
log = nninfo.logger.get_logger(__name__)

# Canonical download/cache directory for torchvision datasets (the repository
# root under an editable install). Tasks used to pass root="../", which
# resolves relative to the *current working directory*, so notebooks (run from
# e.g. tmlr_scripts/) and scripts (run from the repo root) ended up with
# different dataset copies. An absolute path makes every entry point share one.
DATASET_ROOT = str(Path(__file__).parents[2])

_QMNIST_LABEL_FILES = {
    "train": "qmnist-train-labels-idx2-int",
    "test": "qmnist-test-labels-idx2-int",
    "test10k": "qmnist-test-labels-idx2-int",
    "test50k": "qmnist-test-labels-idx2-int",
}
_QMNIST_ROW_SLICES = {
    "train": slice(None),
    "test": slice(None),
    "test10k": slice(None, 10000),
    "test50k": slice(10000, None),
}


def qmnist_class_labels(qmnist_dataset):
    """Class labels of a torchvision QMNIST dataset, robust to a parsing bug.

    Some torchvision builds (e.g. 0.14 on Apple Silicon) mis-parse the
    big-endian idx2-int QMNIST label file and return garbage (typically an
    all-zero class column). If the parsed labels look invalid, re-read them
    from the raw file with an explicit big-endian dtype.
    """
    labels = qmnist_dataset.targets[:, 0]
    if _labels_valid(labels):
        return labels

    log.warning(
        "torchvision returned invalid QMNIST class labels; "
        "re-parsing raw label file with explicit big-endian dtype."
    )
    label_file = (
        Path(qmnist_dataset.root)
        / "QMNIST"
        / "raw"
        / _QMNIST_LABEL_FILES[qmnist_dataset.what]
    )
    raw = np.fromfile(label_file, dtype=">i4")
    n, cols = int(raw[1]), int(raw[2])
    all_labels = raw[3:].reshape(n, cols)[:, 0].astype(np.int64)
    labels = torch.from_numpy(all_labels[_QMNIST_ROW_SLICES[qmnist_dataset.what]].copy())

    assert len(labels) == len(qmnist_dataset.targets)
    assert _labels_valid(labels), "Raw QMNIST label file could not be parsed"
    return labels


def _labels_valid(labels):
    return (
        labels.min() >= 0
        and labels.max() <= 9
        and len(torch.unique(labels)) == 10
    )

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