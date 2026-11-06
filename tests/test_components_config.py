"""Config round-trip tests for Trainer, Tester, TaskManager and the
ExperimentComponent parent mechanism."""

from types import SimpleNamespace

import pytest

from nninfo.exp_comp import ExperimentComponent
from nninfo.tasks.task_manager import TaskManager
from nninfo.tester import Tester as NetworkTester  # alias: avoid pytest collection
from nninfo.tester import dict_zip
from nninfo.trainer import Trainer


class TestTrainerConfig:
    def make_trainer(self):
        return Trainer(
            dataset_name="full_set/train",
            optim_str="SGD",
            loss_str="BCELoss",
            lr=0.01,
            shuffle=True,
            batch_size=64,
            quantizer=[None, {"levels": 8, "rounding_point": "stochastic"}, None],
            momentum=0.9,
        )

    def test_round_trip(self):
        trainer = self.make_trainer()
        restored = Trainer.from_config(trainer.to_config())
        assert restored.to_config() == trainer.to_config()

    def test_from_config_defaults(self):
        # older configs may not carry momentum/quantizer/regularizer keys
        minimal = {
            "dataset_name": "full_set/train",
            "optim_str": "Adam",
            "loss_str": "CELoss",
            "lr": 0.001,
            "shuffle": False,
            "batch_size": 32,
        }
        trainer = Trainer.from_config(minimal)
        config = trainer.to_config()
        assert config["momentum"] == 0
        assert config["quantizer"] is None
        assert config["reg_str"] is None

    def test_counters_start_at_zero(self):
        trainer = self.make_trainer()
        assert trainer.n_epochs_trained == 0
        assert trainer.n_chapters_trained == 0


class TestTesterConfig:
    def test_round_trip(self):
        tester = NetworkTester(dataset_name="full_set/test")
        restored = NetworkTester.from_config(tester.to_config())
        assert restored.to_config() == tester.to_config()


class TestDictZip:
    def test_zips_by_key(self):
        a = {"x": 1, "y": 2}
        b = {"x": 10, "y": 20}
        assert dict(dict_zip(a, b)) == {"x": (1, 10), "y": (2, 20)}


class TestTaskManagerConfig:
    def test_round_trip_with_split(self):
        tm = TaskManager("tishby_dat")
        tm["full_set"].train_test_val_random_split(2796, 1300, 0, seed=42)

        restored = TaskManager.from_config(tm.to_config())

        assert restored.to_config() == tm.to_config()
        assert len(restored["full_set/train"]) == 2796
        # same seed -> same shuffling on reload
        assert (
            restored["full_set/train"].indices == tm["full_set/train"].indices
        ).all()

    def test_binning_limits_lookup(self):
        tm = TaskManager("tishby_dat")
        assert tm.get_binning_limits("X") == "binary"
        assert tm.get_binning_limits("Y") == "binary"
        with pytest.raises(AttributeError):
            tm.get_binning_limits("L1")

    def test_input_output_dimensions(self):
        tm = TaskManager("tishby_dat")
        assert tm.get_input_output_dimensions() == (12, 1)


class TestExperimentComponentParent:
    def test_initial_parent_assignment(self):
        comp = ExperimentComponent()
        parent = SimpleNamespace(id="exp1")
        comp.parent = parent
        assert comp.parent is parent

    @pytest.mark.xfail(
        raises=IndexError,
        reason="known bug: malformed log call on parent reassignment — "
        "'.format(type(self))' misses the second placeholder and parent.id is "
        "passed to log.warning as an extra arg (nninfo/exp_comp.py:26)",
        strict=True,
    )
    def test_parent_reassignment(self):
        comp = ExperimentComponent()
        comp.parent = SimpleNamespace(id="exp1")
        new_parent = SimpleNamespace(id="exp2")
        comp.parent = new_parent
        assert comp.parent is new_parent
