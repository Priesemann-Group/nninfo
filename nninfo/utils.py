import shutil

from .experiment import Experiment


def remove_experiment(experiment_id, yes=False, silent=False):
    """
    Removes an experiment from the database.

    Args:
        experiment_id (str): id of the experiment to be removed
        yes (bool): if True, no confirmation is asked
        silent (bool): if True, no error is raised if experiment does not exist
    """
    experiment_dir = Experiment._find_experiment_dir(experiment_id)

    if not experiment_dir.exists():
        if silent:
            print(
                f'Experiment {experiment_id} does not exist at path {experiment_dir}. Skipping removal.')
            return
        raise ValueError(
            f'Experiment {experiment_id} does not exist at path {experiment_dir}.')

    if not yes:
        print(
            f'Are you sure you want to remove experiment {experiment_id} at path {experiment_dir}? (y/n)')
        answer = input()
        if answer != 'y':
            return

    # Recursively remove experiment directory
    shutil.rmtree(experiment_dir)

    print(f'Removed experiment {experiment_id} from database.')
