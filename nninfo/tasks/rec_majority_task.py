from .task import Task

class RecMajorityTask(Task):
    task_id = "rec_maj"

    def __init__(self, **kwargs):
        """
        Expected kwargs:
            voter_list: list of numbers of voters for each recursive layer
        """
        super().__init__(self, kwargs)

        assert "voter_list" in kwargs

    def x_limits(self):
        return "binary"

    def y_limits(self):
        return "binary"

    def x_dim(self):
        return self._kwargs["voter_list"][0]

    def y_dim(self):
        return 1

    def generate_sample(self, rng):
        x = rng.integers(1, size=10)