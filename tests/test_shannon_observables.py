"""Tests for nninfo.postprocessing.shannon_observables."""

import numpy as np
import pandas as pd
import pytest

from nninfo.postprocessing.shannon_observables import (
    add_shannon_observables,
    per_source_vulnerability,
)


def make_results(n_sources=3, total_MI=2.0, source_MI=None, allbut=None):
    """A deg_red result frame with hand-chosen mutual informations."""
    source_MI = source_MI if source_MI is not None else [1.0] * n_sources
    allbut = allbut if allbut is not None else [total_MI] * n_sources
    row = {"run_id": 0, "chapter_id": 1, "epoch_id": 10, "total_MI": total_MI}
    row.update({f"source_MI{i}": v for i, v in enumerate(source_MI)})
    row.update({f"allbut_source_MI{i}": v for i, v in enumerate(allbut)})
    return pd.DataFrame([row])


class TestAddShannonObservables:
    def test_fully_redundant_code(self):
        # each source alone gives all the information; nothing is critical
        df = add_shannon_observables(
            make_results(3, total_MI=2.0, source_MI=[2.0] * 3,
                         allbut=[2.0] * 3))
        assert df["deg_red"].iloc[0] == pytest.approx(3.0)
        assert df["deg_vuln"].iloc[0] == pytest.approx(0.0)

    def test_fully_unique_code(self):
        # 3 independent sources of 2/3 bit each: removing one loses its share
        df = add_shannon_observables(
            make_results(3, total_MI=2.0, source_MI=[2 / 3] * 3,
                         allbut=[4 / 3] * 3))
        assert df["deg_red"].iloc[0] == pytest.approx(1.0)
        assert df["deg_vuln"].iloc[0] == pytest.approx(1.0)

    def test_maximally_vulnerable_code(self):
        # XOR-like: no single source informative, every source critical
        df = add_shannon_observables(
            make_results(3, total_MI=1.0, source_MI=[0.0] * 3,
                         allbut=[0.0] * 3))
        assert df["deg_red"].iloc[0] == pytest.approx(0.0)
        assert df["deg_vuln"].iloc[0] == pytest.approx(3.0)

    def test_rejects_entropy_only_frames(self):
        df = pd.DataFrame([{"run_id": 0, "chapter_id": 0, "epoch_id": 0,
                            "total_H": 1.0, "source_H0": 0.5}])
        with pytest.raises(ValueError):
            add_shannon_observables(df)

    def test_does_not_mutate_input(self):
        df = make_results()
        cols = list(df.columns)
        add_shannon_observables(df)
        assert list(df.columns) == cols


class TestPerSourceVulnerability:
    def test_contributions_sum_to_deg_vuln(self):
        res = make_results(3, total_MI=2.0, source_MI=[1.0] * 3,
                           allbut=[1.5, 1.8, 1.9])
        total = add_shannon_observables(res)["deg_vuln"].iloc[0]
        per = per_source_vulnerability(res)
        assert per["v_i"].sum() == pytest.approx(total)

    def test_neuron_numbering_is_one_based(self):
        per = per_source_vulnerability(make_results(3))
        assert sorted(per["neuron"]) == [1, 2, 3]
