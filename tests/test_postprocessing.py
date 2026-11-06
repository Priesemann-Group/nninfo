"""Tests for PID postprocessing (nninfo/postprocessing/pid_postprocessing.py)."""

import numpy as np
import pandas as pd
import pytest

from nninfo.postprocessing.pid_postprocessing import (
    Antichain,
    get_pid_summary_quantities,
)


class TestAntichain:
    def test_parses_tuple_string(self):
        a = Antichain("((1,), (2,))")
        assert len(a) == 2
        assert list(a) == [(1,), (2,)]

    def test_degree_of_synergy_is_min_source_set_size(self):
        assert Antichain("((1,), (2,))").degree_of_synergy() == 1  # redundancy
        assert Antichain("((1,),)").degree_of_synergy() == 1  # unique
        assert Antichain("((1, 2),)").degree_of_synergy() == 2  # synergy
        assert Antichain("((1, 2), (3,))").degree_of_synergy() == 1

    def test_max_index(self):
        assert Antichain("((1, 2),)").max_index() == 2
        assert Antichain("((1,), (3,))").max_index() == 3

    def test_rejects_non_antichain_strings(self):
        with pytest.raises(AssertionError):
            Antichain("(1, 2)")  # not a tuple of tuples


def make_pid_frame():
    """Two-source PID rows the way PIDMeasurement.save_results lays them out."""
    atoms = {
        "((1,), (2,))": [0.2, 0.1],  # redundancy  (degree 1)
        "((1,),)": [0.3, 0.2],  # unique 1    (degree 1)
        "((2,),)": [0.1, 0.2],  # unique 2    (degree 1)
        "((1, 2),)": [0.4, 0.5],  # synergy     (degree 2)
    }
    part = pd.DataFrame(atoms)
    pid = pd.concat(
        [part, part, part], axis=1, keys=["inf_pid", "misinf_pid", "avg_pid"]
    )
    pid.insert(0, "run_id", 0)
    pid.insert(1, "chapter_id", [0, 1])
    pid.insert(2, "epoch_id", [0, 10])
    return pid


class TestSummaryQuantities:
    def test_mutual_information_is_atom_sum(self):
        summary = get_pid_summary_quantities(make_pid_frame())
        mi = np.ravel(summary["mutual_information"].to_numpy())
        assert mi == pytest.approx([1.0, 1.0])

    def test_degree_of_synergy_atoms(self):
        summary = get_pid_summary_quantities(make_pid_frame())
        atoms = summary["degree_of_synergy_atoms"]
        assert np.ravel(atoms[1].to_numpy()) == pytest.approx([0.6, 0.5])
        assert np.ravel(atoms[2].to_numpy()) == pytest.approx([0.4, 0.5])

    def test_representational_complexity(self):
        # sum_k k * atoms_k / MI: row0 -> 1*0.6 + 2*0.4 = 1.4; row1 -> 0.5 + 1.0 = 1.5
        summary = get_pid_summary_quantities(make_pid_frame())
        rc = np.ravel(summary["representational_complexity"].to_numpy())
        assert rc == pytest.approx([1.4, 1.5])

    def test_ids_carried_through(self):
        summary = get_pid_summary_quantities(make_pid_frame())
        assert np.ravel(summary["chapter_id"].to_numpy()).tolist() == [0, 1]
        assert np.ravel(summary["epoch_id"].to_numpy()).tolist() == [0, 10]
