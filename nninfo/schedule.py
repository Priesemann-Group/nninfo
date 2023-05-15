
import numpy as np

import nninfo


class Schedule:
    """
    Can create epoch lists for preplanned experiment chapters. These chapters are the main
    structure of the training period of the experiment and allow for spaced saving of
    checkpoints.

    The plan is to end a chapter of the experiment when a epoch contained in the chapter_ends
    variable of this class is reached. This is not applied yet, but the class
    is already able to create log spaced and lin spaced numbers of epochs,
    which should then help with the actual experiment run. However, for the log-spaced chapter
    planning, the number of chapter_ends can be lower than the number of chapters that are
    given as n_chapter_wished.

    Does not need to inherit from ExperimentComponent, because it is not calling anything else.
    """

    def __init__(self):
        self.chapter_ends = None
        self.chapter_ends_continued = None

    @staticmethod
    def from_config(config):
        """
        Creates a Schedule object from a config dictionary.

        Args:
            config (dict): Dictionary containing the config for the Schedule object.

        Returns:
            Schedule object.
        """

        schedule = Schedule()
        schedule.chapter_ends = config["chapter_ends"]
        schedule.chapter_ends_continued = config["chapter_ends_continued"]
        return schedule

    def to_config(self):
        """
        Creates a config dictionary from the Schedule object.

        Returns:
            Dictionary containing the config for the Schedule object.
        """

        config = {
            "chapter_ends": self.chapter_ends,
            "chapter_ends_continued": self.chapter_ends_continued,
        }
        return config

    def create_log_spaced_chapters(self, n_epochs, n_chapters_wished):
        """
        Function that creates a list of numbers which are the epoch indices where chapters
        are ended. The indices are created logarithmically spaced over the total number of
        epochs for this experiment (n_epochs).

        Args:
            n_epochs (int): Total number of epochs for this experiment.
            n_chapters_wished (int): Number of chapters the experiment should take to reach the
                total number of epochs n_epochs.

        Sets self.chapter_ends to a list of these indices (int).

        Sets self.chapter_ends_continued to a list of a continued of chapter_ends
        until n_epochs*n_epochs (int).
        """

        def log_space(n_e, n_c):
            end = np.log10(n_e)
            epochs = np.logspace(0, end, n_c + 1, endpoint=True)
            epochs = np.round(epochs).astype(int)
            epochs = np.unique(epochs)
            # add a 0 in the front for consistency
            epochs = np.insert(epochs, 0, 0)
            return epochs

        self.chapter_ends = log_space(n_epochs, n_chapters_wished).tolist()
        self.chapter_ends_continued = log_space(
            n_epochs * n_epochs, n_chapters_wished * 2
        ).tolist()

    def create_lin_spaced_chapters(self, n_epochs, n_chapters_wished):
        """
        Function that creates a list of numbers, which are the epoch indices where chapters
        are ended. The indices are created linearly spaced over the total number of
        epochs for this experiment (n_epochs).

        Args:
            n_epochs (int): Total number of epochs for this experiment.
            n_chapters_wished (int): Number of chapters the experiment should take to reach the
                total number of epochs n_epochs.

        Sets self.chapter_ends to a list of these indices (int).

        Sets self.chapter_ends_continued to a list of a continued of chapter_ends
        until n_epochs*100 (int).
        """

        def lin_space(n_e, n_c):
            epochs = np.linspace(0, n_e, n_c + 1, endpoint=True)
            epochs = np.round(epochs).astype(int)
            epochs = np.unique(epochs)
            return epochs

        self.chapter_ends = lin_space(n_epochs, n_chapters_wished).tolist()
        self.chapter_ends_continued = lin_space(
            n_epochs * 100, n_chapters_wished * 100
        ).tolist()

    def get_epoch_for_chapter(self, chapter):
        """
        Returns the epoch index for a given chapter index.

        Args:
            chapter (int): Index of the chapter.

        Returns:
            Epoch index (int).
        """

        return self.chapter_ends_continued[chapter]

    def save(self, path):
        saver = nninfo.file_io.FileManager(path, write=True)
        save_dict = {
            "chapter_ends": self.chapter_ends,
            "chapter_ends_continued": self.chapter_ends_continued,
        }
        saver.write(save_dict, "schedule.json")

    def _load(self, path):
        loader = nninfo.file_io.FileManager(path, read=True)
        load_dict = loader.read("schedule.json")
        self.chapter_ends = load_dict["chapter_ends"]
        self.chapter_ends_continued = load_dict["chapter_ends_continued"]

    def __str__(self):
        return str(self.chapter_ends)
