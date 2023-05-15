# you may not need all of these, if you know your data file types
# and the way FileManager handles them.
import os, re, glob
from pathlib import Path
import shutil
import pickle
import json
import ast
import copy

import numpy as np
import scipy.io as io
import torch
import yaml
import pandas as pd
from filelock import FileLock

import nninfo
from .exp_comp import ExperimentComponent

FILELOCK_TIMEOUT = -1


log = nninfo.logger.get_logger(__name__)

EXPERIMENT_DIR_STANDARD = {
    "experiment_dir": "exp_{:04d}/",
    "experiment_dir_str": "exp_{}/",
    "checkpoints": "checkpoints/",
    "measurements": "measurements/",
    "plots": "plots/",
    "logs": "log/",
    "components": "components/",
}

FILENAME_STANDARD = {
    "checkpoint": "ckpt_r{:06d}_c{:06d}_e{:012d}.pt",
    "checkpoint_loading": "ckpt_r{:06d}_c{:06d}_e*.pt",
    "checkpoint_loading_all_runs": "ckpt_r*_c{:06d}_e*.pt",
    "checkpoint_loading_all_chapters": "ckpt_r{:06d}_c*_e*.pt",
    "measurement_history": "measurement_history.json",
    "measurement_file": "meas_r{:06d}_c{:06d}_e{:012d}.jsonl",
}

CHECKPOINT_FILE_NAME_PATTERN = "ckpt_r*_c*_e*.pt"

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


class CheckpointManager(ExperimentComponent):
    """
    Implemented following this:
    https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3

    Allows for storing and reloading of the network state,
    optimizer state and random number generator states.
    """

    def __init__(self, checkpoint_dir):
        super().__init__()
        self._checkpoint_dir = checkpoint_dir

    def save(self, filename=None):
        state = {
            "chapter_id": self.parent.chapter_id,
            "model_state_dict": self.parent.network.state_dict(),
            "optimizer_state_dict": self.parent.trainer.optimizer_state_dict(),
            "epoch_id": self.parent.epoch_id,
            "torch_seed": torch.get_rng_state(),
            "numpy_seed": np.random.get_state(),
            "run_id": self.parent.run_id,
        }
        """
        Saves the current state of the experiment as a checkpoint
        in the experiments/expXXXX/checkpoints/
        directory. Outputs a message if succeeded.
        """

        if filename is None:
            filename = FILENAME_STANDARD["checkpoint"]

        final_filename = filename.format(
            self.parent.run_id, self.parent.chapter_id, self.parent.epoch_id
        )
        torch.save(state, self._checkpoint_dir / final_filename)

        log.info(
            "Successfully saved current state of the training as {}.".format(
                final_filename
            )
        )
    
    def get_checkpoint_filename(self, run_id, chapter_id):
        """Returns the filename of the checkpoint with the given run_id and chapter_id.
        
        Args:
            run_id (int): run_id of the checkpoint
            chapter_id (int): chapter_id of the checkpoint
            
        Returns:
            str: filename of the checkpoint
        """
        
        file_name = FILENAME_STANDARD["checkpoint_loading"].format(run_id, chapter_id)
        
        file_paths = list(self._checkpoint_dir.glob(file_name))

        if len(file_paths) == 0:
            raise FileNotFoundError(f"Could not find file {file_name}")
        elif len(file_paths) > 1:
            raise RuntimeError(f"Found more than one file matching {file_name}")
        
        file_path = file_paths[0]
        
        return file_path
    
    def get_checkpoint(self, run_id, chapter_id):
        """Returns the checkpoint with the given run_id and chapter_id.
        
        Args:
            run_id (int): run_id of the checkpoint
            chapter_id (int): chapter_id of the checkpoint
            
        Returns:
            dict: checkpoint
        """
        
        file_name = self.get_checkpoint_filename(run_id, chapter_id)

        return torch.load(file_name)

    def read(self, filename):
        if self._checkpoint_loader is None:
            self.init_loader_saver()
        return self._checkpoint_loader.read(filename)

    def list_all_checkpoints(self):
        """Collects all checkpoint files in the checkpoints directory.
        
        Returns:
            list: list of tuples (run_id, chapter_id)
        """

        # Get all checkpoint files that match the pattern
        checkpoints = self._checkpoint_dir.glob(CHECKPOINT_FILE_NAME_PATTERN)
        checkpoints = [checkpoint.name for checkpoint in checkpoints]

        # extract run_id and chapter_id from filename
        checkpoints = [
            (
                int(re.findall(r"r(\d+)", checkpoint)[0]),
                int(re.findall(r"c(\d+)", checkpoint)[0]),
            )
            for checkpoint in checkpoints
        ]

        return sorted(checkpoints)

    def list_checkpoints(self, run_ids=None, chapter_ids=None):
        
        checkpoints = self.list_all_checkpoints()

        if run_ids is not None:

            if isinstance(run_ids, int):
                run_ids = [run_ids]

            checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint[0] in run_ids]

        if chapter_ids is not None:

            if isinstance(chapter_ids, int):
                chapter_ids = [chapter_ids]

            checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint[1] in chapter_ids]

        return checkpoints
    
    def get_run_ids(self):
        """Returns a sorted list of all runs that have checkpoints."""
        return sorted(list(set(checkpoint[0] for checkpoint in self.list_all_checkpoints())))
    
    def get_chapter_ids(self, run_id):
        """Returns a sorted list of all chapters that have checkpoints."""
        return sorted(list(set(checkpoint[1] for checkpoint in self.list_all_checkpoints() if checkpoint[0] == run_id)))
    
    def get_last_run(self):
        """Returns the last run that has checkpoints."""

        runs = self.get_run_ids()

        if len(runs) == 0:
            raise ValueError("No runs found.")
        
        return max(runs)
    
    def get_last_chapter_in_run(self, run_id):
        """Returns the last chapter in a run that has a checkpoint."""

        checkpoints = self.list_checkpoints(run_ids=[run_id])

        if len(checkpoints) == 0:
            raise ValueError("No checkpoints found for run_id {}".format(run_id))

        return max(checkpoints, key=lambda x: x[1])[1]
        

class MeasurementManager:
    """
    Allows for storing and reloading of the network state,
    optimizer state and random number generator states.
    """

    def __init__(self, experiment_dir: Path, measurement_subdir="measurements/"):
        self._measurement_saver = FileManager(
            experiment_dir / measurement_subdir, write=True
        )
        self._measurement_loader = FileManager(
            experiment_dir / measurement_subdir, read=True
        )
        self._history_dict = None
        lock = FileLock(FILENAME_STANDARD["measurement_history"]+".lock", timeout=FILELOCK_TIMEOUT)
        with lock.acquire():
            self.load_history()

    def load_history(self):
        try:
            self._history_dict = self._measurement_loader.read(
                "measurement_history.json"
            )
        except FileNotFoundError:
            log.info(
                "Measurement history not found. Creating new measurement_history.json"
            )
            self._measurement_saver.write(dict(), "measurement_history.json")
            self._history_dict = self._measurement_loader.read(
                "measurement_history.json"
            )

    def get_next_measurement_id(self):
        # This function could also look into the settings of old measurements in order not to
        # repeat parts of or full measurements. At the moment it just finds the maximum of the
        # other ids though
        try:
            id = int(max(self._history_dict, key=int)) + 1
        except Exception as e:
            id = 0
        return id

    def save(self, measurement, measurement_id=None):
        """
        Save a measurement by appending it to the right file.
        """
        lock = FileLock(FILENAME_STANDARD["measurement_history"]+".lock", timeout=FILELOCK_TIMEOUT)
        with lock.acquire():
            self.load_history()

            if measurement_id is None:
                measurement_id = self.get_next_measurement_id()
            save_dict = measurement.get_measurement_dict(measurement_id)

            run_id = save_dict["run_id"]
            chapter_id = save_dict["chapter_id"]
            epoch_id = save_dict["epoch_id"]
            type_str = save_dict["measurement_type"]

            if type_str == "pid":
                file_prefix = "active_"
            elif type_str == "fisher":
                file_prefix = "structural_"
            elif type_str == "weight":
                file_prefix = "structural_"
            elif type_str == "mi":
                file_prefix = "active_"
            else:
                raise NotImplementedError
            self._measurement_saver.append(
                save_dict,
                file_prefix
                + FILENAME_STANDARD["measurement_file"].format(
                    run_id, chapter_id, epoch_id
                ),
            )
            if measurement_id in self._history_dict:
                self._history_dict[measurement_id].append((run_id, chapter_id, epoch_id))
            else:
                self._history_dict[measurement_id] = [(run_id, chapter_id, epoch_id)]
            self._measurement_saver.write(
                self._history_dict, FILENAME_STANDARD["measurement_history"]
            )

        return measurement_id

    def load(self, filename="*.jsonl", **kwargs):
        file_list = self._measurement_loader.list_files_in_dir(filename)
        row_list = []
        for f in file_list:
            row_list.extend(self._measurement_loader.read(f, **kwargs))
        return pd.DataFrame(row_list)

    @property
    def history_dict(self):
        self.load_history()
        return copy.deepcopy(self._history_dict)


class FileManager:
    def __init__(self, rel_path, read=False, write=False):
        if (read and write) or (not read and not write):
            log.error("FileManager should either read or write in one directory.")
            raise AttributeError

        self._read = read
        self._write = write
        module_dir = os.path.dirname(__file__)
        self._rel_path = rel_path
        self._path = os.path.join(module_dir, rel_path)

        self.tuple_as_str = False

    def list_subdirs_in_dir(self):
        d = self._path
        return [
            os.path.join(d, o)
            for o in os.listdir(d)
            if os.path.isdir(os.path.join(d, o))
        ]

    def list_files_in_dir(self, filename=None):
        if filename is not None:
            filepath = os.path.join(self._path, filename)
            names = [os.path.basename(x) for x in glob.glob(filepath)]
            return names
        return os.listdir(self._path)

    def find_file(self, filename):
        # here glob is used, because it is nicer for a filename that includes a '*'
        filepath = os.path.join(self._path, filename)
        filepath_list = glob.glob(filepath)
        if len(filepath_list) == 1:
            new_filepath = filepath_list[0]
        elif len(filepath_list) > 1:
            log.warning("Multiple files meet this criterion: {}".format(filepath_list))
            idx = input("Specify index (0,1..)")
            log.info("Chose index {}".format(idx))
            new_filepath = filepath_list[idx]
        else:
            raise FileNotFoundError
        return os.path.basename(new_filepath)

    def write(self, data, filename):
        if not self.write:
            log.error("Not allowed to write.")
            raise PermissionError

        ext = os.path.splitext(filename)[1]
        if ext == ".pt":
            self._write_torch_state_pt(data, filename)
        elif ext == ".jsonl":
            self._write_jsonl(data, filename)
        elif ext == ".json":
            self._write_json(data, filename)
        elif ext == ".npy":
            self._write_npy(data, filename)
        elif ext == ".yaml":
            self._write_yaml(data, filename)
        else:
            raise NotImplementedError

    def append(self, data, filename):
        if not self.write:
            log.error("Not allowed to write.")
            raise PermissionError

        ext = os.path.splitext(filename)[1]
        if ext == ".jsonl":
            self._write_jsonl(data, filename, append=True)
        else:
            log.error("Appending not implemented for file extension " + ext)
            raise NotImplementedError

    def _write_torch_state_pt(self, state, filename):
        filepath = os.path.join(self._path, filename)
        torch.save(state, filepath)

    def _write_jsonl(self, data, filename, append=False):
        filepath = os.path.join(self._path, filename)
        enc = MultiDimensionalArrayEncoder()
        json_str = enc.encode(data)
        if append:
            with open(filepath, "a") as f:
                f.write(json_str + "\n")
        else:
            with open(filepath, "w") as f:
                f.write(json_str + "\n")

    def _write_json(self, data, filename):
        filepath = os.path.join(self._path, filename)
        enc = MultiDimensionalArrayEncoder()
        json_str = enc.encode(data)
        with open(filepath, "w") as f:
            f.write(json_str)

    def _write_npy(self, data, filename):
        filepath = os.path.join(self._path, filename)
        np.save(filepath, data, allow_pickle=False)

    def _write_yaml(self, data, filename):
        filepath = os.path.join(self._path, filename)
        with open(filepath, "w") as f:
            yaml.dump(data, f, Dumper=NoAliasDumper, sort_keys=False)

    def read(self, filename, line=None, **kwargs):
        if not self.read:
            log.error("Permission Error: Not allowed to read.")
            raise PermissionError

        ext = os.path.splitext(filename)[1]
        if ext == ".npy":
            return self._read_numpy(filename)
        elif ext == ".mat":
            return self._read_mat(filename)
        elif ext == ".pt":
            return self._read_torch_state_pt(filename)
        elif ext == ".pkl":
            return self._read_pickle(filename)
        elif ext == ".jsonl":
            return self._read_jsonl(filename, line, **kwargs)
        elif ext == ".json":
            return self._read_json(filename, **kwargs)
        elif ext == ".yaml":
            return self._read_yaml(filename, **kwargs)
        else:
            log.error("Filetype {} not supported.".format(ext))
            raise IOError

    def _read_torch_state_pt(self, filename):
        filepath = os.path.join(self._path, filename)
        return torch.load(filepath)

    def _read_numpy(self, filename):
        filepath = os.path.join(self._path, filename)
        return np.load(filepath)

    def _read_mat(self, filename):
        filepath = os.path.join(self._path, filename)
        return io.loadmat(filepath)

    def _read_pickle(self, filename):
        filepath = os.path.join(self._path, filename)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data

    def _read_jsonl(self, filename, jsonl_line_id=None, **kwargs):
        filepath = os.path.join(self._path, filename)
        if "tuple_as_str" in kwargs:
            self.tuple_as_str = kwargs["tuple_as_str"]
        else:
            self.tuple_as_str = True
        if jsonl_line_id is None:
            row_list = []
            with open(filepath, "r") as f:
                for line in f:
                    if line.startswith("{"):
                        json_line = json.loads(line, object_hook=self.hinted_tuple_hook)
                        row_list.append(json_line)
            return row_list
        else:
            json_string = ""
            with open(filepath, "r") as f:
                for i, line in enumerate(f):
                    if i == jsonl_line_id - 1:
                        json_string = line.rstrip("\n")
                        break
            return json.loads(json_string, object_hook=self.hinted_tuple_hook)

    def _read_json(self, filename, **kwargs):
        if "tuple_as_str" in kwargs:
            self.tuple_as_str = kwargs["tuple_as_str"]
        else:
            self.tuple_as_str = False
        filepath = os.path.join(self._path, filename)
        with open(filepath, "r") as f:
            return json.load(f, object_hook=self.hinted_tuple_hook)
        
    def _read_yaml(self, filename, **kwargs):
        filepath = os.path.join(self._path, filename)
        with open(filepath, "r") as f:
            return yaml.safe_load(f)

    def make_experiment_dir(self, id, overwrite=False):
        if not self.write:
            log.error("Not allowed to write.")
            raise PermissionError
        experiment_dir = EXPERIMENT_DIR_STANDARD["experiment_dir_str"].format(id)

        experiment_dir_abs_path = os.path.join(self._path, experiment_dir)
        experiment_dir_rel_path = os.path.join(self._rel_path, experiment_dir)

        if os.path.exists(experiment_dir_abs_path):
            log.warning(
                "Experiment directory with same id already exists: "
                + experiment_dir_abs_path
            )
            if overwrite:
                log.warning("Overwrite existing experiment directory?")
                if input("Overwrite existing experiment directory? [y/n]") == "y":
                    #log.info(
                    #    "Overwriting experiment directory with id {} accepted by user.".format(
                    #        id
                    #    )
                    #)
                    shutil.rmtree(experiment_dir_abs_path)
                else:
                    log.error("Overwriting not permitted by user.")
                    raise PermissionError
            else:
                log.error("Overwriting not set as argument when calling Experiment().")
                raise PermissionError

        os.mkdir(experiment_dir_abs_path)

        for key, value in EXPERIMENT_DIR_STANDARD.items():
            if not key == "experiment_dir" and not key  == "experiment_dir_str":
                new_dir = os.path.join(experiment_dir_abs_path, value)
                os.mkdir(new_dir)

        log.info(
            "Successfully made new experiment directory {} for experiment {}".format(
                experiment_dir_rel_path, str(id)
            )
        )
        return Path(experiment_dir_abs_path)

    def hinted_tuple_hook(self, obj):
        """
        Helper function that is called for decoding of json lines files.
        Whenever the json object that is
        encoded has strings as key or value arguments that are strings, they are checked
        for the prefix '__tuple__'. If such a tuple is found, the rest of the string
        is decoded via ast.literal_eval and thereby turned into a tuple.
        While this works fine for smaller files, it
        might cause a significant slowdown for larger files. However, this does not hurt
        too much the performance, since we are just appending to the measurement
        files. Also, this is why in nninfo the
        standard is, to have several json objects in one file, each on a seperate line.
        Also keep in mind the potential security risk of ast.literal_eval.
        """

        def parse(item):
            if isinstance(item, str):
                if item.startswith("__tuple__"):
                    if self.tuple_as_str:
                        parsed = item.lstrip("__tuple__")
                    else:
                        parsed = ast.literal_eval(item.lstrip("__tuple__"))
                elif item.startswith("__int__"):
                    parsed = ast.literal_eval(item.lstrip("__int__"))
                else:
                    parsed = item
            elif isinstance(item, list):
                parsed = list()
                for el in item:
                    parsed.append(parse(el))
            elif isinstance(item, dict):
                parsed = dict()
                for k, v in item.items():
                    parsed_key = parse(k)
                    parsed_value = parse(v)
                    parsed[parsed_key] = parsed_value
            else:
                parsed = item
            return parsed

        return parse(obj)


class MultiDimensionalArrayEncoder(json.JSONEncoder):
    """
    Encoder, inherits from json.JSONEncoder. Modifies it in the sense that it handles
    tuples differently from lists. Tuples are packed into strings before writing to to
    the file. Each of these str(tuple()) objects has a prefix "__tuple__" which identifies
    it as a tuple. When loading the json object, these strings are decoded via the
    hinted_tuple_hook.

    Keyword arguments can be used to set the underlying JSONEncoder objects at __init__.
    For example, skipkeys, ensure_ascii, check_circular, allow_nan, sort_keys, indent can
    be set. However, for the functionality of nninfo, the JSON objects have to be in one
    line each.
    """

    def __init__(self, **kwargs):
        super(MultiDimensionalArrayEncoder, self).__init__(**kwargs)

    def encode(self, obj):
        def hint_tuples(item, key=False):
            if isinstance(item, tuple):
                return "__tuple__" + str(item)
            if isinstance(item, int) and key:
                return "__int__" + str(item)
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {
                    hint_tuples(key, key=True): hint_tuples(value)
                    for key, value in item.items()
                }
            else:
                return item

        return super(MultiDimensionalArrayEncoder, self).encode(hint_tuples(obj))
