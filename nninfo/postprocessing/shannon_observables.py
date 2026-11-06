"""Shannon observables from `deg_red` PIDMeasurement results.

A `PIDMeasurement` with ``pid_definition='deg_red'`` stores, per checkpoint,
the total mutual information of a layer with the target together with the
mutual informations of each individual source and of each leave-one-out
group. This module turns those raw columns into the two Shannon observables
of the paper:

* average degree of redundancy  :math:`\\bar r = \\sum_i I(X_i;Y) / I(\\bm X;Y)`
* average degree of vulnerability :math:`\\bar v = \\sum_j I(X_j;Y\\mid \\bm X_{-j}) / I(\\bm X;Y)`

Note that :math:`\\bar v` is *not* a degree of synergy: synergy belongs to the
:math:`\\bar r` perspective (atoms with :math:`r=0`), while :math:`\\bar v`
counts the sources a typical piece of information critically depends on.
"""

import pandas as pd

__all__ = [
    "add_shannon_observables",
    "per_source_vulnerability",
    "load_shannon_observables",
]

SOURCE_MI = r"^source_MI[0-9]+$"
ALLBUT_MI = r"^allbut_source_MI[0-9]+$"


def add_shannon_observables(results: pd.DataFrame) -> pd.DataFrame:
    """Add ``deg_red`` and ``deg_vuln`` columns to a `deg_red` result frame.

    Args:
        results: DataFrame as returned by ``PIDMeasurement.results`` for a
            measurement with ``pid_definition='deg_red'``.

    Returns:
        A copy with two extra columns.
    """
    src = results.filter(regex=SOURCE_MI, axis=1)
    allbut = results.filter(regex=ALLBUT_MI, axis=1)
    if src.empty or allbut.empty:
        raise ValueError(
            "No source_MI/allbut_source_MI columns found -- is this a "
            "'deg_red' measurement (and not run with entropy_only=True)?"
        )
    n_sources = allbut.shape[1]

    results = results.copy()
    results["deg_red"] = src.sum(axis=1) / results["total_MI"]
    results["deg_vuln"] = (
        n_sources * results["total_MI"] - allbut.sum(axis=1)
    ) / results["total_MI"]
    return results


def per_source_vulnerability(results: pd.DataFrame) -> pd.DataFrame:
    """Long-format per-source contributions to the degree of vulnerability.

    The contribution of source ``i`` is
    :math:`v_i = I(X_i;Y\\mid \\bm X_{-i}) / I(\\bm X;Y)`, so that
    :math:`\\bar v = \\sum_i v_i`.

    Returns:
        DataFrame with columns run_id, chapter_id, epoch_id, neuron (1-based),
        v_i, and the raw total_MI.
    """
    n_sources = results.filter(regex=ALLBUT_MI, axis=1).shape[1]
    rows = []
    for _, r in results.iterrows():
        for i in range(n_sources):
            rows.append(
                {
                    "run_id": int(r["run_id"]),
                    "chapter_id": int(r["chapter_id"]),
                    "epoch_id": int(r["epoch_id"]),
                    "neuron": i + 1,  # 1-based, matching NeuronID
                    "v_i": (r["total_MI"] - r[f"allbut_source_MI{i}"])
                    / r["total_MI"],
                    "total_MI": r["total_MI"],
                }
            )
    return pd.DataFrame(rows)


def load_shannon_observables(experiment, layer, measurement_id=None):
    """Load one layer's `deg_red` measurement and add the observables.

    Args:
        experiment: Experiment instance.
        layer (int): layer label number, e.g. 4 for "L4".
        measurement_id (str, optional): defaults to
            ``summary_measurement_L{layer}_stochastic``.
    """
    measurement_id = measurement_id or f"summary_measurement_L{layer}_stochastic"
    path = experiment.experiment_dir / measurement_id / "results.h5"
    return add_shannon_observables(pd.read_hdf(path, "results"))
