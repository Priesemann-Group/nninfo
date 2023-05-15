import torch
import numpy as np
from numpy.random import Philox, Generator

SUBSET_SYMBOL = "/"  # <dataset>/<subset>/<subsubset> etc.

class DataSet(torch.utils.data.Dataset):
    def __init__(self, task, name):
        self._task = task
        self._name = name
        self._subsets = []

    @staticmethod
    def from_config(task, config):
        """
        Creates a new DataSet from a config dictionary
        """
        if task.finite:
            return CachedDataset.from_config(task, config)
        else:
            return LazyDataset.from_config(task, config)
        
    def to_config(self):
        """
        Returns a dictionary representation of the dataset tree
        """
        d = dict(name=self._name)
        
        if self._subsets:
            d["subsets"] = [subset.to_config() for subset in self._subsets]

        return d
    
    def _load_subsets_from_config_list(self, config_list):
        self._subsets = [
            SubSet.from_config(self, subset_config)
            for subset_config in config_list
        ]

    def __str__(self, level=0):
        """
        Recursive function that allows for printing of the Dataset Tree / Subset Branch.

        Args:
            level (int): level of branch

        Returns:
            str: Representation of this branch (Tree).
        """
        ret = "\t" * level + self.__repr__() + "\n"
        for subset in self._subsets:
            ret += subset.__str__(level=level + 1)
        return ret

    def __repr__(self):
        return (
            self._name
            + ": \t"
            + str(len(self))
            + " elements."
            + ("(lazy)" if isinstance(self, LazyDataset) else "")
            + ("(cached)" if isinstance(self, CachedDataset) else "")
        )

    def find(self, dataset_name):
        """
        Depth-first search for dataset_name in the dataset tree
        """
        if self._name == dataset_name:
            return self
        else:
            for subset in self._subsets:
                result = subset.find(dataset_name)
                if not result is None:
                    return result
            return None

    def create_subset(self, name, subset_slice, random_split_seed=None):
        subset = SubSet(self, name, subset_slice, random_split_seed=random_split_seed)
        self._subsets.append(subset)

    def train_test_val_random_split(self, train_len, test_len, val_len, seed):

        total_len = train_len + test_len + val_len

        if len(self) != total_len:
            raise ValueError(
                "Split can only be performed if the subdatasets total"
                "length matches with the length of the dataset"
            )

        train_name = self._name + SUBSET_SYMBOL + "train"
        test_name = self._name + SUBSET_SYMBOL + "test"
        val_name = self._name + SUBSET_SYMBOL + "val"

        # create subsets
        self.create_subset(train_name, slice(None, train_len), random_split_seed=seed)
        self.create_subset(test_name, slice(train_len, train_len + test_len), random_split_seed=seed)
        self.create_subset(val_name, slice(train_len + test_len, None), random_split_seed=seed)
        return [train_name, test_name, val_name]

    def train_test_val_sequential_split(self, train_len, test_len, val_len):
        assert train_len + test_len + val_len == len(self), 'Split can only be performed if the subsets comprise the whole set'

        train_name = self._name + SUBSET_SYMBOL + "train"
        test_name = self._name + SUBSET_SYMBOL + "test"
        val_name = self._name + SUBSET_SYMBOL + "val"

        self.create_subset(train_name, slice(None, train_len))
        self.create_subset(test_name, slice(train_len, train_len + test_len))
        self.create_subset(val_name, slice(train_len + test_len, None))

    def one_class_split(self):
        return self._class_wise_split("one_class")

    def all_but_one_class_split(self):
        return self._class_wise_split("all_but_one_class")

    def multiple_class_split(self, class_list):
        return self._class_wise_split("multiple", class_list=class_list)

    def _class_wise_split(self, method, class_list=None):
        dataset_labels_np = self._y# if self is CachedDataset else np.array(self)[:, 1]
        # TODO: Test this for one-hot-labels
        classes = np.unique(dataset_labels_np)
        return_name_list = []
        if method == "one_class":
            for cls_idx, cls in enumerate(classes):
                keep_idx = np.where(dataset_labels_np == cls)[0]
                cls_name = self._name + SUBSET_SYMBOL + "class_" + str(cls_idx)
                self.create_subset(cls_name, keep_idx)
                return_name_list.append(cls_name)
        elif method == "all_but_one_class":
            for cls_idx, cls in enumerate(classes):
                keep_idx = np.where(dataset_labels_np != cls)[0]
                cls_name = self._name + SUBSET_SYMBOL + "not_class_" + str(cls_idx)
                self.create_subset(cls_name, keep_idx)
                return_name_list.append(cls_name)
        elif method == "multiple":
            keep_idx_list = []
            cls_name = self._name + SUBSET_SYMBOL + "multiple"
            for el in class_list:
                temp = np.where(dataset_labels_np == el)[0].tolist()
                print(type(temp))
                keep_idx_list.extend(temp)
                cls_name += "_" + str(el)
            keep_idx = np.array(keep_idx_list)
            self.create_subset(cls_name, keep_idx)
        else:
            raise NotImplementedError
        return return_name_list

    @property
    def name(self):
        return self._name

    @property
    def task(self):
        return self._task

    @property
    def subsets(self):
        return self._subsets


class SubSet(DataSet):
    """
    SubSet of data, that is directly connected to the original 'full_set', and takes its samples
    from there. Never change the order of the parent Dataset!
    """

    def __init__(self, parent, name, subset_slice, random_split_seed=None):
        """Defines a subset of the parent dataset, that is defined by the indices in parent_indices

        Args:
            parent (DataSet): Parent dataset
            name (str): Name of the subset
            subset_slice (slice): Slice of the parent indices that are used for this subset
            random_split_seed (int, optional): Seed for shuffling the parent indices before slicing
        """
        super().__init__(parent.task, name)
        self._parent = parent
        self._random_split_seed = random_split_seed
        self._slice = subset_slice

        if random_split_seed is not None:
            generator = np.random.default_rng(random_split_seed)
            parent_indices = generator.permutation(len(parent))
        else:
            parent_indices = np.arange(len(parent))

        self._indices = parent_indices[subset_slice]

    @staticmethod
    def from_config(parent, config):
        
        subsets = [SubSet.from_config(parent, subset_config) for subset_config in config.get('subsets', [])]

        subset = SubSet(parent, config['name'], eval(config['subset_slice']), config.get('random_split_seed', None))
        subset._subsets = subsets

        return subset
    
    def to_config(self):

        d = dict(name=self._name, subset_slice=str(self._slice))

        if self._random_split_seed is not None:
            d['random_split_seed'] = self._random_split_seed

        if self._subsets:
            d['subsets'] = [subset.to_dict() for subset in self._subsets]

        return d

    def __getitem__(self, idx):
        return self.parent[self._indices[idx]]

    def __len__(self):
        return len(self._indices)

    @property
    def parent(self):
        return self._parent

    @property
    def indices(self):
        return self._indices


class CachedDataset(DataSet):
    def __init__(self, task, name):
        super().__init__(task, name)
        x, y = self._task.load_samples()
        self._x, self._y = x, y

    @staticmethod
    def from_config(task, config):
        dataset = CachedDataset(task, config['name'])
        dataset._load_subsets_from_config_list(config.get('subsets', []))
        return dataset

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]


class LazyDataset(DataSet):
    def __init__(self, task, name, length, seed=None, condition=None):
        super().__init__(task, name)
        assert not task.finite
        self._len = length

        if not seed is None:
            self._seed = seed
        else:
            seed_generator = Generator(Philox())
            self._seed = int(
                seed_generator.integers(np.iinfo(np.int64).max)
            )  # Has to be int to be json serializable

        self._philox = Philox(seed)
        self._philox_state = self._philox.state
        self._rng = Generator(self._philox)

        self._condition = condition

    @staticmethod
    def from_config(task, config):
        dataset = LazyDataset(
            task, config['name'], config['length'], config.get('seed', None), config.get('condition', None)
        )

        dataset._load_subsets_from_config_list(config.get('subsets', []))

        return dataset
    
    def to_config(self):
        d = dict(
            name=self._name,
            length=self._len,
            seed=self._seed,
            condition=self._condition,
        )

        if self._subsets:
            d["subsets"] = [subset.to_dict() for subset in self._subsets]

        return d
    
    def extend(self, n_samples):
        """
        Get a copy of the dataset with a different length
        """
        name_ext = self._name + "_ext"
        return LazyDataset(self._task, name_ext, n_samples, self._seed)

    def condition(self, n_samples, condition):
        """
        Get a copy of the dataset with a condition
        """
        name_cond = self._name + "_cond"
        return LazyDataset(
            self._task, name_cond, n_samples, self._seed, condition=condition
        )

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx >= self._len:
            raise IndexError

        self._philox_state["state"]["counter"][-1] = idx
        self._philox.state = self._philox_state
        return self._task.generate_sample(self._rng, condition=self._condition)
    