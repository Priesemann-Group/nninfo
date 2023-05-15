from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import filelock
import pandas as pd
import yaml

from ..file_io import NoAliasDumper
from ..experiment import Experiment


class Measurement(ABC):
    MEASUREMENT_CONFIG_FILE_NAME = "config.yaml"
    MEASUREMENT_RESULTS_FILE_NAME = "results.h5"
    MEASUREMENT_RESULTS_FILE_LOCK = "results.lock"

    measurement_type: ClassVar[str]

    def __init__(self,
                 experiment: Experiment,
                 measurement_id: str,
                 dataset_name: str,
                 dataset_kwargs: dict = None,
                 quantizer_params: list[dict] = None,
                 _load: bool = False):

        self._experiment = experiment
        self._measurement_id = measurement_id
        self._quantizer_params = quantizer_params

        self._dataset_name = dataset_name
        self._dataset_kwargs = dataset_kwargs or {}

        self._measurement_dir = experiment.experiment_dir / measurement_id
        self._config_file = self._measurement_dir / self.MEASUREMENT_CONFIG_FILE_NAME

        self._results_file = self._measurement_dir / self.MEASUREMENT_RESULTS_FILE_NAME
        self._results_file_lock = self._measurement_dir / \
            self.MEASUREMENT_RESULTS_FILE_LOCK

        if not _load:
            self._create_dir_and_config()

    def _create_dir_and_config(self):

        try:
            self._measurement_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if self._config_file.exists():
                config = Measurement._load_yaml(self._config_file)
                if config == self.to_config():
                    return
            raise FileExistsError(
                "The measurement directory already exists and the config does not match.")

        with open(self._config_file, "w") as f:
            yaml.dump(self.to_config(), f, Dumper=NoAliasDumper)

    def to_config(self):
        return {
            "measurement_type": self.measurement_type,
            "experiment_id": self._experiment.id,
            "measurement_id": self._measurement_id,
            "dataset_name": self._dataset_name,
            "dataset_kwargs": self._dataset_kwargs,
            "quantizer_params": self._quantizer_params
        }

    @classmethod
    def from_config(cls, experiment, config):

        config = config.copy()

        assert config.pop("measurement_type") == cls.measurement_type, \
            "The measurement type in the config does not match the measurement type of the class."
        assert config.pop("experiment_id") == experiment.id, \
            "The experiment id in the config does not match the experiment id of the experiment."

        return cls(experiment=experiment, **config, _load=True)

    @staticmethod
    def _load_yaml(file):
        with open(file, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def load(cls, experiment: Experiment, measurement_id: str):
        config_file = experiment.experiment_dir / \
            measurement_id / cls.MEASUREMENT_CONFIG_FILE_NAME

        config = cls._load_yaml(config_file)

        return cls.from_config(experiment, config)

    def save_results(self, run_id, chapter_id, results_df):
        """Appends the results to the hdf5 file in the corresponding measurement directory.

        """

        # Add columns for run_ids, chapter_ids and epoch_ids
        results_df.insert(loc=0, column="run_id", value=run_id)
        results_df.insert(loc=1, column="chapter_id", value=chapter_id)
        results_df.insert(loc=2, column="epoch_id",
                            value=self._experiment.schedule.get_epoch_for_chapter(chapter_id))

        with filelock.FileLock(self._results_file_lock):
            with pd.HDFStore(self._results_file) as store:
                store.put("results", results_df,
                          append=True, format="table")

    def perform_measurements(self, run_ids: Literal['all'] | int | list[int], chapter_ids: Literal['all'] | int | list[int], exists_ok: bool = True):

        checkpoint_ids = self._experiment.checkpoint_manager.list_all_checkpoints()

        if not run_ids == 'all':

            if isinstance(run_ids, int):
                run_ids = [run_ids]

            # Filter out checkpoint_ids that are not in run_ids
            checkpoint_ids = [c for c in checkpoint_ids if c[0] in run_ids]

        if not chapter_ids == 'all':

            if isinstance(chapter_ids, int):
                chapter_ids = [chapter_ids]

            # Filter out checkpoint_ids that are not in chapter_ids
            checkpoint_ids = [c for c in checkpoint_ids if c[1] in chapter_ids]

        print(
            f"Performing measurements for {len(checkpoint_ids)} checkpoints: {checkpoint_ids}.")

        # Perform measurements for all run_ids and chapter_ids sequentially
        for checkpoint_id in checkpoint_ids:
            self.perform_measurement(checkpoint_id[0], checkpoint_id[1], exists_ok=exists_ok)

    def perform_measurement(self, run_id: int, chapter_id: int, exists_ok: bool = True):

        if not self._results_file.exists():
            self._create_results_file()

        # Check if there is already a result for this run_id and chapter_id
        if self._check_if_result_exists(run_id, chapter_id):
            if exists_ok:
                print(
                    f"There is already a {self.measurement_type!r} result for run_id {run_id} and chapter_id {chapter_id}. Skipping.")
                return
            raise ValueError(
                f"There is already a {self.measurement_type!r} result for run_id {run_id} and chapter_id {chapter_id}. Aborting.")

        print(
            f"Performing {self.measurement_type!r} measurement for run_id {run_id} and chapter_id {chapter_id}.")
        results = self._measure(run_id, chapter_id)

        self.save_results(run_id, chapter_id, results)

    def _create_results_file(self):
        """Creates the results file.

        """
        with filelock.FileLock(self._results_file_lock):
            with pd.HDFStore(self._results_file) as store:
                store.put("results", pd.DataFrame(),
                          append=True, format="table")

    def _check_if_result_exists(self, run_id: int, chapter_id: int):

        results = self.results
        if results[(results["run_id"] == run_id) & (results["chapter_id"] == chapter_id)].shape[0] > 0:
            return True

    @abstractmethod
    def _measure(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def results(self):

        with filelock.FileLock(self._results_file_lock):
            with pd.HDFStore(self._results_file) as store:
                try:
                    results_df = store.select("results")
                except KeyError:
                    results_df = pd.DataFrame(columns=["run_id", "chapter_id", "epoch_id"])

        return results_df
