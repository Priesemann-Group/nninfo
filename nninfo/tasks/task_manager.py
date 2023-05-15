from ..exp_comp import ExperimentComponent
from ..data_set import DataSet, CachedDataset, LazyDataset
from .task import Task

class TaskManager(ExperimentComponent):
    """
    Helper class, handles a task with one 'full_set' dataset and several subsets,
    which are then used by Trainer, Tester or Evaluation.
    """

    def __init__(self, task_id=None, **kwargs):
        """
        Creates a new instance of TaskManager. Loads the
        dataset according to task_id.

        Keyword Args:
            task_id (str): one of the pre-implemented Tasks:
                'tishby_dat', 'andreas_dat', 'fake_dat' etc.

        Passes additional arguments on to dataset creation if given.
        """

        super().__init__()

        self._dataset = None
        self._kwargs = kwargs

        task_kwargs = kwargs.get("task_kwargs", {})
        self.task = Task.from_id(task_id, **task_kwargs)

        if self.task.finite:
            self._dataset = CachedDataset(self.task, "full_set")
        else:
            seed = kwargs["seed"]
            n_samples = kwargs["n_samples"]
            self._dataset = LazyDataset(self.task, "full_set", n_samples, seed)
    
    @staticmethod
    def from_config(config):
        """
        Creates a new TaskManager from a config dictionary
        """
        task_manager = TaskManager(
            task_id=config["task_id"],
            **config["kwarg_dict"],
        )

        task_manager._dataset = DataSet.from_config(task_manager.task, config["subsets"])

        return task_manager

    def to_config(self):
        """
        Creates a config dictionary from the TaskManager
        """
        output_dict = {
            "task_id": self.task.task_id,
            "kwarg_dict": self._kwargs,
            "subsets": self._dataset.to_config(),
            "task_kwargs": self.task.kwargs,
        }

        return output_dict

    def get_binning_limits(self, label):
        if label == "X":
            return self.task.x_limits
        elif label == "Y":
            return self.task.y_limits
        else:
            raise AttributeError

    def get_input_output_dimensions(self):
        return self.task.x_dim, self.task.y_dim

    def __getitem__(self, dataset_name):
        """
        Finds the dataset by the given name in the dataset tree
        """
        return self._dataset.find(dataset_name)

    def __str__(self, level=0):
        ret = self._dataset.__str__()
        return ret