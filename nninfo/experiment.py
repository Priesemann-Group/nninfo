import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

import nninfo
from nninfo.data_set import DataSet
from nninfo.trainer import Trainer
from nninfo.schedule import Schedule
from nninfo.model.neural_network import NeuralNetwork, NeuronID, NoisyNeuralNetwork
from nninfo.tasks.task_manager import TaskManager
from nninfo.tester import Tester
from nninfo.file_io import FileManager, CheckpointManager


log = nninfo.logger.get_logger(__name__)

CONFIG_FILE_NAME = "config.yaml"

class Experiment:
    """
    Manages the entire experiment, is directly in contact with the user script but also
    the main components.
    After connecting the components, user script should use preferably methods of this class.

    1) is given an instance of TaskManager to feed data into the program and split the dataset
       if necessary

    2) is given an instance of NeuralNetwork in model.py that will be trained and tested on

    3) is given an instance of Trainer that is responsible
       for each chapter (predefined set of training epochs)
       of the training process (gets data from TaskManager via dataset_name).

    4) is given an instance of Tester that is called after each chunk of training is done
       (gets data from TaskManager)

    5) can be given an instance of Schedule. This makes automating the experiment easier.

    6) creates an instance of CheckpointManager that stores the main experiment parameters
       for loading afterwards. This can be used to
       a) analyze the training afterwards
       b) resume training from last or earlier chapters
    """

    def __init__(self,
                 experiment_id: str,
                 network: NeuralNetwork,
                 task: TaskManager,
                 trainer: Trainer,
                 tester: Tester,
                 schedule: Schedule,
                 _load: bool = False):
        """Creates a new Experiment instance from the given components.

        Args:
            experiment_id (str): Unique identifier for the experiment.
            network (NeuralNetwork): NeuralNetwork instance.
            task (TaskManager): TaskManager instance.
            trainer (Trainer): Trainer instance.
            tester (Tester): Tester instance.
            schedule (Schedule): Schedule instance.
            _load (bool, optional): Internal flag only. For loading an experiment, use Experiment.load().
        """

        self._experiment_id = experiment_id
        self._run_id = 0
        self._experiment_dir = self._find_experiment_dir(experiment_id)

        if not _load:

            if os.path.exists(self._experiment_dir):
                raise FileExistsError(f"Experiment directory {self._experiment_dir} already exists."
                                      "Please choose a different experiment_id or use Experiment.load(experiment_id).")

            standard_dir_maker = FileManager(
                "../experiments/", write=True
            )
            self._experiment_dir = standard_dir_maker.make_experiment_dir(
                experiment_id, overwrite=False)

        nninfo.logger.add_exp_file_handler(self._experiment_dir)
        log.info(f"Starting exp_{experiment_id}")

        # create checkpoint_manager that saves checkpoints to _experiment_dir for the entire experiment
        self.checkpoint_manager = CheckpointManager(self._experiment_dir / "checkpoints")
        self.checkpoint_manager.parent = self

        self._set_components(network, task, trainer, tester, schedule)

    @staticmethod
    def _find_experiment_dir(experiment_id: str):
        """Finds the experiment directory based on the experiment id.

        Args:
            experiment_id (str): Unique identifier for the experiment.

        Returns:
            str: Path to the experiment directory.
        """

        return Path(__file__).parent.parent / "experiments" / f"exp_{experiment_id}"
    
    @staticmethod
    def load(exp_id: str):
        """Loads an experiment from a file.

        Args:
            exp_id (str): Unique identifier for the experiment.

        Returns:
            Experiment: Experiment instance.
        """

        experiment_dir = Experiment._find_experiment_dir(exp_id)
        return Experiment.load_file(experiment_dir / CONFIG_FILE_NAME)

    @staticmethod
    def load_file(path: Path):
        """Loads an experiment from a file.

        Args:
            path (Path): Path to the experiment file.

        Returns:
            Experiment: Experiment instance.
        """
        file_manager = FileManager(path.parent, read=True)
        config = file_manager.read(path.name)
        return Experiment.from_config(config)

    @staticmethod
    def from_config(config):
        """Loads an experiment from a config dict.

        Args:
            config (dict): Dictionary containing the experiment configuration.

        Returns:
            Experiment: Experiment instance.
        """

        experiment_id = config["experiment_id"]
        network = NeuralNetwork.from_config(config["network"])
        task = TaskManager.from_config(config["task"])
        trainer = Trainer.from_config(config["trainer"])
        tester = Tester.from_config(config["tester"])
        schedule = Schedule.from_config(config["schedule"])

        experiment = Experiment(experiment_id, network,
                                task, trainer, tester, schedule, _load=True)
        trainer.initialize_components()
        experiment.load_last_checkpoint()

        return experiment

    def to_config(self):
        """Creates a config dictionary from the experiment.

        Returns:
            dict: Dictionary containing the experiment configuration.
        """

        config = {
            "experiment_id": self._experiment_id,
            "network": self._network.to_config(),
            "task": self._task.to_config(),
            "trainer": self._trainer.to_config(),
            "tester": self._tester.to_config(),
            "schedule": self._schedule.to_config()
        }

        return config

    def load_last_checkpoint(self):
        """Loads the last checkpoint of the experiment."""

        try:
            last_run = self.checkpoint_manager.get_last_run()
        except ValueError:
            raise

        try:
            last_chapter = self.checkpoint_manager.get_last_chapter_in_run(last_run)
        except ValueError:
            raise

        self.load_checkpoint(last_run, last_chapter)

    def load_checkpoint(self, run_id, chapter_id):
        """Loads a checkpoint of the experiment.

        Args:
            run_id (int): Run id of the checkpoint.
            chapter_id (int): Chapter id of the checkpoint.
        """

        checkpoint = self.checkpoint_manager.get_checkpoint(run_id, chapter_id)

        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.trainer.load_optimizer_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        self.trainer.set_n_chapters_trained(checkpoint["chapter_id"])
        self.trainer.set_n_epochs_trained(checkpoint["epoch_id"])

        torch.set_rng_state(checkpoint["torch_seed"])
        np.random.set_state(checkpoint["numpy_seed"])

        log.info(f"Successfully loaded checkpoint {run_id}-{chapter_id}.")

        self._run_id = checkpoint["run_id"]

    def save_components(self):

        component_saver = FileManager(
            self._experiment_dir, write=True)

        config = self.to_config()
        component_saver.write(config, CONFIG_FILE_NAME)

    def run_following_schedule(self, continue_run=False, chapter_ends=None, use_cuda=False, use_ipex=False, compute_test_loss=None):

        if chapter_ends is None:
            if self._schedule is None:
                log.error(
                    "You can only use run_following_schedule if you have "
                    + "a schedule connected to the experiment or pass a schedule."
                )
                return
            else:
                chapter_ends = self._schedule.chapter_ends

        if continue_run:
            log.warning(
                "Continuing run {} at chapter {}.".format(
                    self.run_id, self.chapter_id)
            )
        else:
            if self.chapter_id != 0 or self.epoch_id != 0:
                log.error(
                    "You can only use run_following_schedule if you reset the training to a new run."
                )
                return

        info = "Starting training on run {} starting at chapter {}, epoch {}".format(
            self.run_id, self.chapter_id, self.epoch_id
        )
        log.info(info)
        print(info)
        for c in range(self.chapter_id, len(chapter_ends) - 1):
            if chapter_ends[c] != self.epoch_id:
                log.error(
                    "Error on continuing schedule,"
                    + " schedule.chapter_ends[{}]={}".format(c, chapter_ends[c])
                    + " and experiment's self.epoch_id={} ".format(self.epoch_id)
                    + "do not fit together."
                )
                raise ValueError
            # running c+1:
            n_epochs_chapter = chapter_ends[c + 1] - chapter_ends[c]
            self._trainer.train_chapter(
                n_epochs_chapter=n_epochs_chapter, use_cuda=use_cuda, use_ipex=use_ipex, compute_test_loss=compute_test_loss)

    def continue_runs_following_schedule(self, runs_id_list, stop_epoch, schedule=None, use_cuda=False, compute_test_loss=None):
        if schedule is None:
            if self._schedule is None:
                log.error(
                    "You can only use run_following_schedule if you have "
                    + "a schedule connected to the experiment or pass a schedule."
                )
                return
            else:
                schedule = self._schedule

        cut_off = np.argmax(
            np.array(schedule.chapter_ends_continued) > stop_epoch)
        chapter_ends = schedule.chapter_ends_continued[:cut_off]
        for run_id in runs_id_list:
            last_chapter = self.checkpoint_manager.get_last_chapter_in_run(run_id)
            self.load_checkpoint(run_id, last_chapter)
            self.run_following_schedule(
                continue_run=True, chapter_ends=chapter_ends, use_cuda=use_cuda, compute_test_loss=compute_test_loss)

    def rerun(self, n_runs, like_run_id=None):
        """
        Reruns the experiment for a given number of runs. For doing this, it uses
        the checkpoints of a previous run that are found in the checkpoints directory
        and produces the same checkpoints with a new network initialization.

        Args:
            n_runs (int): Number of additional runs that should be performed.
            like_run_id (int): Run id of the run that should be replicated. If not set,
                defaults to run_id=0.
        """
        log.info("Setting up rerun of experiment: n_runs=" + str(n_runs))

        if like_run_id is None:
            like_run_id = 0

        ckpt_list = self.checkpoint_manager.list_checkpoints(run_ids=[
                                                              like_run_id])

        ckpt_list = [ckpt[1] for ckpt in ckpt_list]
        epoch_list = [self.schedule.get_epoch_for_chapter(c) for c in ckpt_list]
        log.warning("Extracted schedule: " + str(epoch_list))

        last_run_id = self.checkpoint_manager.get_last_run()

        for i in range(n_runs):
            # getting everything to the same state as requested
            self.load_checkpoint(run_id=like_run_id, chapter_id=0)
            # get the new run_id
            self._run_id = last_run_id + 1
            # reinitialize the network with a new seed
            self._network.init_weights(randomize_seed=True)
            self.run_following_schedule(chapter_ends=epoch_list)
            last_run_id = self._run_id

    def save_checkpoint(self):
        """
        Calls the CheckpointManager to save the current state of the network and the optimizer
        (is necessary for optimizers that depend on their own past)
        together with the state of random number generators (of numpy and torch).
        """
        self.checkpoint_manager.save()

    def _set_components(self, network, task, trainer, tester, schedule):
        self._network = network
        self._network.parent = self

        self._task = task
        self._task.parent = self

        self._trainer = trainer
        self._trainer.parent = self

        self._tester = tester
        self._tester.parent = self

        self._schedule = schedule
        self._schedule.parent = self

    @property
    def all_key_components_connected(self):
        """
        Property (function that is disguised as an object variable)
        that checks whether all components for this experiment
        are already in place. (For now, all are needed to start the experiment,
        this could be changed in the future though, for example Test might not
        be relevant for every experiment.)

        Returns:
             (bool): All components are connected, True or False.
        """

        all_comp_flag = True
        if self._task is None:
            log.info("Task still missing.")
            all_comp_flag = False
        if self._network is None:
            log.info("Network still missing.")
            all_comp_flag = False
        if self._trainer is None:
            log.info("Trainer still missing.")
            all_comp_flag = False
        if self._tester is None:
            log.info("Tester still missing.")
            all_comp_flag = False
        return all_comp_flag
    
    def capture_activations(
            self,
            dataset,
            run_id,
            chapter_id,
            repeat_dataset=1,
            batch_size=10 ** 4,
            before_noise=False,
            condition=None,
            quantizer_params=None,
        ):
        """
        Captures activations for current network state for the given dataset.

        Returns:
            Iterator that yields dictionaries with keys X, Y, L1, L2, ...
                and ndarrays of size (batch_size, ) containing activations
        """

        # Load checkpoint for given run and chapter
        self.load_checkpoint(run_id, chapter_id)

        assert isinstance(self.network, NoisyNeuralNetwork) or before_noise == False, 'Extraction after noise only possible for noisy network!'
        
        if not isinstance(dataset, DataSet):
            dataset = self.task[dataset]

        if not condition is None:
            dataset = dataset.condition(len(dataset), condition)
        
        feeder = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        self.network.eval()
        
        for x, y in feeder:
            for _ in range(repeat_dataset):

                if isinstance(self.network, NoisyNeuralNetwork):
                    act_dict = self.network.extract_activations(
                        x, before_noise=before_noise, quantizer_params=quantizer_params
                    )
                else:
                    act_dict = self.network.extract_activations(
                        x, quantizer_params=quantizer_params
                    )

                act_dict["X"] = x.detach().numpy()

                act_dict["Y"] = y.detach().numpy()

                # Add decision function. Index of maximum value of output layer. If multiple output neurons have the same activation, choose the first!
                act_dict["Yhat"] = np.argmax(
                    act_dict["L{}".format(len(act_dict) - 2)], axis=1
                )

                # Reshape to neuron id dicts NeuronID->[activations]
                act_dict = {
                    NeuronID(layer_id, (neuron_idx + 1,)): act_dict[layer_id][:, neuron_idx]
                    if act_dict[layer_id].ndim > 1
                    else act_dict[layer_id]
                    for layer_id in act_dict
                    for neuron_idx in range(
                        act_dict[layer_id].shape[1]
                        if act_dict[layer_id].ndim > 1
                        else 1
                    )
                }

                yield act_dict 

    @property
    def experiment_dir(self):
        return self._experiment_dir

    @property
    def id(self):
        return self._experiment_id

    @property
    def run_id(self):
        return self._run_id

    @property
    def chapter_id(self):
        return self.trainer.n_chapters_trained

    @property
    def epoch_id(self):
        return self.trainer.n_epochs_trained

    @property
    def network(self):
        return self._network

    @property
    def trainer(self):
        return self._trainer

    @property
    def task(self):
        return self._task

    @property
    def tester(self):
        return self._tester

    @property
    def schedule(self):
        return self._schedule
