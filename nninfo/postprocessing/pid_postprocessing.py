from ast import literal_eval
from itertools import chain, combinations

import pandas as pd


class Antichain:

    def __init__(self, antichain_string):
        self._antichain_tuple = literal_eval(antichain_string)

        assert isinstance(self._antichain_tuple, tuple)
        assert all(isinstance(a, tuple) for a in self._antichain_tuple)
        assert all(isinstance(l, int)
                   for a in self._antichain_tuple for l in a)

    def degree_of_synergy(self):
        return min(len(a) for a in self._antichain_tuple)
    
    def max_index(self):
        return max(max(a) for a in self._antichain_tuple)

    def __len__(self):
        return len(self._antichain_tuple)

    def __iter__(self):
        return iter(self._antichain_tuple)


def _compute_degree_of_synergy_atoms(pid: pd.DataFrame):

    degrees_of_synergy = [Antichain(antichain_string).degree_of_synergy()
                          for antichain_string in pid.columns]
    degree_of_synergy_atoms = pid.groupby(
        degrees_of_synergy, axis=1).sum()

    return degree_of_synergy_atoms


def _compute_mutual_information(pid: pd.DataFrame):

    mutual_information = pid.sum(axis=1)
    mutual_information.name = ''

    return mutual_information


def _compute_representational_complexity(degree_of_synergy_atoms: pd.DataFrame, mutual_information: pd.DataFrame):

    repr_compl = (degree_of_synergy_atoms *
                  degree_of_synergy_atoms.columns).sum(axis=1) / mutual_information
    repr_compl.name = ''

    return repr_compl


def get_pid_summary_quantities(pid: pd.DataFrame, pid_part: str = 'avg_pid'):
    """
    Compute summary quantities of a PID DataFrame.
    """

    # Select the part of the PID to be used
    pid_part = pid[pid_part]

    degree_of_synergy_atoms = _compute_degree_of_synergy_atoms(pid_part)
    mutual_information = _compute_mutual_information(pid_part)

    repr_compl = _compute_representational_complexity(
        degree_of_synergy_atoms, mutual_information)

    # Combine into a DataFrame with a MultiIndex including run_id, chapter_id and epoch_id

    summary_quantities = degree_of_synergy_atoms
    summary_quantities.columns = pd.MultiIndex.from_arrays(
        [['degree_of_synergy_atoms'] * len(degree_of_synergy_atoms.columns), degree_of_synergy_atoms.columns])
    summary_quantities['mutual_information'] = mutual_information
    summary_quantities['representational_complexity'] = repr_compl
    summary_quantities['run_id'] = pid['run_id']
    summary_quantities['chapter_id'] = pid['chapter_id']
    summary_quantities['epoch_id'] = pid['epoch_id']

    return summary_quantities


def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def get_mi_or_entropy(pid: pd.DataFrame, pid_part: str = 'avg_pid'):

    n = max(Antichain(antichain_string).max_index()
            for antichain_string in pid.avg_pid.columns)

    pid_part = pid[pid_part]
    source_sets = list(_powerset(range(1, n + 1)))[1:]

    df_entropies = pd.DataFrame()
    for source_set in source_sets:
        H = pd.Series(0, index=pid_part.index)
        for achain_str in pid_part.columns:
            achain = Antichain(achain_str)
            for a in achain:
                if set(a).issubset(source_set):
                    H += pid_part[achain_str]
                    break
        df_entropies[str(source_set)] = H
    return df_entropies