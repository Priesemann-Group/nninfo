import torch

from ..file_io import FileManager
from .task import Task

class TishbyTask(Task):
    task_id = "tishby_dat"

    @property
    def finite(self):
        return True

    @property
    def x_limits(self):
        return "binary"

    @property
    def y_limits(self):
        return "binary"

    @property
    def x_dim(self):
        return 12

    @property
    def y_dim(self):
        return 1

    def load_samples(self):
        self._data_location = "../data/Tishby_2017/"
        dataset_reader = FileManager(
            self._data_location, read=True)
        data_dict = dataset_reader.read("var_u.mat")
        x = data_dict["F"]
        y = data_dict["y"].T
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)