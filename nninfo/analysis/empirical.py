"""Shannon observables computed directly from an activation array.

`PIDMeasurement(pid_definition='deg_red')` is the normal route to the
observables, but it always measures the *intact* code stored in a checkpoint.
To characterise a code that has been modified on the fly -- e.g. a layer with
some neurons lesioned -- it is easier to compute the observables straight from
the activations. `code_observables` does that, using the same plug-in
estimator as the measurement pipeline.
"""

from collections import Counter

import numpy as np

__all__ = ["code_observables"]


def code_observables(activations, labels, neurons=None):
    """Total MI, degree of redundancy and degree of vulnerability of a code.

    Args:
        activations: ndarray (n_samples, n_neurons) of *discrete* activations.
        labels: ndarray (n_samples,) of discrete target values.
        neurons: optional list of column indices to include; defaults to all.
            Use this to evaluate the surviving sub-code after a lesion.

    Returns:
        dict with keys ``total_MI``, ``deg_red``, ``deg_vuln``, ``n_sources``.
        For a single source both ``deg_red`` and ``deg_vuln`` are 1: the
        leave-one-out group is empty and carries no information.
    """
    from sxpid import SxPID

    from .pid_measurement import PIDMeasurement

    a = activations if neurons is None else activations[:, list(neurons)]
    n = a.shape[1]
    if n == 0:
        raise ValueError("need at least one source neuron")

    pmf = Counter(zip(*[zip(col) for col in a.T], zip(labels)))
    total = sum(pmf.values())
    pdf = SxPID.PDF.from_dict({k: v / total for k, v in pmf.items()})

    p = pdf.probs
    p_T = PIDMeasurement.marginalize_same_shape(pdf, *range(n))
    p_S = PIDMeasurement.marginalize_same_shape(pdf, n)
    total_MI = PIDMeasurement._compute_MI(None, p, p, p_S, p_T)

    source_MI, allbut_MI = [], []
    for i in range(n):
        rest = tuple(k for k in range(n) if k != i)
        source_MI.append(
            PIDMeasurement._compute_MI(
                None, p,
                PIDMeasurement.marginalize_same_shape(pdf, *rest),
                PIDMeasurement.marginalize_same_shape(pdf, *(rest + (n,))),
                p_T,
            )
        )
        if n > 1:
            allbut_MI.append(
                PIDMeasurement._compute_MI(
                    None, p,
                    PIDMeasurement.marginalize_same_shape(pdf, i),
                    PIDMeasurement.marginalize_same_shape(pdf, i, n),
                    p_T,
                )
            )
        else:
            allbut_MI.append(0.0)

    deg_red = sum(source_MI) / total_MI if total_MI > 0 else np.nan
    deg_vuln = (
        (n * total_MI - sum(allbut_MI)) / total_MI
        if total_MI > 0 else np.nan
    )
    return {"total_MI": total_MI, "deg_red": deg_red,
            "deg_vuln": deg_vuln, "n_sources": n}
