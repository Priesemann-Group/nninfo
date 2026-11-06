"""Shared test fixtures for the nninfo test suite."""

import pytest

from nninfo.tasks.xor_task import XorTask


class ConcreteXorTask(XorTask):
    """Instantiable XOR task for tests.

    All generative tasks in nninfo (finite=False) currently fail to
    instantiate because ``Task.load_samples`` is an @abstractmethod that they
    do not override (see test_tasks.py::TestKnownBugs). This subclass provides
    the missing override so LazyDataset & sampling logic can still be tested.
    """

    task_id = "xor_test_dat"  # unique id to avoid clobbering the registry entry

    def load_samples(self):
        raise NotImplementedError("Generative task has no finite sample set.")


@pytest.fixture
def xor_task():
    return ConcreteXorTask()
