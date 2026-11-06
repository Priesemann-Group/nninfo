"""Tests for the Task registry and label encoders (nninfo/tasks/)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from nninfo.tasks.task import (
    DATASET_ROOT,
    Task,
    binary_encode_label,
    quaternary_encode_label,
    octal_encode_label,
)
from nninfo.tasks.tishby_task import TishbyTask
from nninfo.tasks.xor_task import XorTask


class TestTaskRegistry:
    def test_subclasses_self_register(self):
        available = Task.list_available_tasks()
        assert "tishby_dat" in available
        assert "xor_dat" in available

    def test_from_id_returns_correct_class(self):
        assert isinstance(Task.from_id("tishby_dat"), TishbyTask)

    def test_from_id_unknown_raises(self):
        with pytest.raises(KeyError):
            Task.from_id("no_such_task")

    def test_generative_task_interface(self, xor_task):
        assert xor_task.finite is False
        assert xor_task.x_dim == 2
        assert xor_task.y_dim == 1
        assert xor_task.x_limits == (0, 1)
        assert xor_task.y_limits == "binary"


class TestXorTaskSamples:
    def test_generate_sample_is_xor(self, xor_task):
        rng = np.random.default_rng(0)
        for _ in range(50):
            x, y = xor_task.generate_sample(rng)
            expected = (x[0] >= 0.5) ^ (x[1] >= 0.5)
            assert y.item() == float(expected)

    def test_finite_task_rejects_generate_sample(self):
        task = Task.from_id("tishby_dat")
        with pytest.raises(NotImplementedError):
            task.generate_sample(np.random.default_rng(0))


class TestTishbyTaskData:
    def test_load_samples_shapes(self):
        x, y = Task.from_id("tishby_dat").load_samples()
        assert x.shape == (4096, 12)  # 2**12 binary input patterns
        assert y.shape == (4096, 1)
        assert set(np.unique(x.numpy())) == {0.0, 1.0}
        assert set(np.unique(y.numpy())) == {0.0, 1.0}


class TestLabelEncoders:
    def test_binary_encode_label(self):
        y = torch.tensor([0, 1, 2, 5])
        bits = binary_encode_label(y, 3)
        expected = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1]]  # LSB first
        )
        assert torch.equal(bits.long(), expected)

    def test_quaternary_encode_label(self):
        y = torch.tensor([0, 1, 4, 9])
        quits = quaternary_encode_label(y, 2)
        expected = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 2]])  # least significant first
        assert torch.equal(quits.long(), expected)

    def test_octal_encode_label_single_digit(self):
        # single-digit case: digit equals the label itself for y < 8
        y = torch.arange(8)
        octs = octal_encode_label(y, 1)
        assert torch.equal(octs.long().flatten(), y)

    @pytest.mark.xfail(
        reason="known bug: octal_encode_label combines PAIRS of quaternary "
        "digits (q0 + 4*q1), producing base-16 digits 0..15 instead of octal "
        "digits 0..7 for multi-digit labels (nninfo/tasks/task.py:90). "
        "combined_mnist_octal_task uses octal_encode_label(y, 2) with labels "
        "up to 19 and is affected.",
        strict=True,
    )
    def test_octal_encode_label_multi_digit(self):
        y = torch.tensor([0, 7, 8, 63])
        octs = octal_encode_label(y, 2)
        expected = torch.tensor([[0, 0], [7, 0], [0, 1], [7, 7]])
        assert torch.equal(octs.long(), expected)

    def test_encoders_invert(self):
        y = torch.arange(64)
        bits = binary_encode_label(y, 6).long()
        reconstructed = (bits * (2 ** torch.arange(6))).sum(dim=1)
        assert torch.equal(reconstructed, y)


class TestDatasetRoot:
    def test_dataset_root_is_absolute_repo_root(self):
        root = Path(DATASET_ROOT)
        assert root.is_absolute()
        assert (root / "nninfo" / "tasks" / "task.py").exists()

    @pytest.mark.skipif(
        not (Path(DATASET_ROOT) / "QMNIST" / "raw").exists(),
        reason="QMNIST data not downloaded",
    )
    def test_mnist1d_test_labels_are_valid(self):
        # regression test for the torchvision QMNIST endianness bug: the 50k
        # QMNIST portion of the test set used to come back with all-zero labels
        _, y = Task.from_id("mnist_1d_dat").load_samples()
        qmnist_portion = y[70000:]  # after 60k MNIST train + 10k MNIST test
        assert qmnist_portion.min() >= 0 and qmnist_portion.max() <= 9
        assert len(torch.unique(qmnist_portion)) == 10
        assert (qmnist_portion == 0).float().mean() < 0.2  # not all zeros


class TestKnownBugs:
    @pytest.mark.xfail(
        raises=TypeError,
        reason="known bug: all generative tasks (xor_dat, parity_dat, "
        "checkerboard_dat, xor_misinfo_dat) fail to instantiate because "
        "Task.load_samples is an @abstractmethod they do not override "
        "(nninfo/tasks/task.py:74)",
        strict=True,
    )
    def test_generative_tasks_instantiable(self):
        task = Task.from_id("xor_dat")
        assert isinstance(task, XorTask)
