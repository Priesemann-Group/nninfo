"""Conventional (non-information-theoretic) neuron importance measures.

These are the standard baselines against which the degree of vulnerability
can be compared: weight magnitudes, activation statistics, and a first-order
Taylor saliency.

A caveat that matters when comparing layers: the incoming and outgoing weight
vectors of a neuron have as many components as the adjacent layers are wide,
so raw L2 norms of neurons in layers with different fan-in/fan-out are *not*
on a common scale. ``weight_importances`` therefore also returns
dimension-corrected (``*_rms``) and within-layer relative (``*_rel``)
variants; only the latter is dimensionless and comparable between layers.
"""

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["weight_importances", "activation_importances", "taylor_saliency",
           "collect_importances"]


def weight_importances(network, layer_idx):
    """Incoming/outgoing weight norms for every neuron of one layer.

    The module list skips layers without parameters, so ``layer_idx`` (an
    index into ``layer_infos``) is mapped to module indices by counting the
    parametrised layers before it.

    Returns:
        dict of 1-D numpy arrays, one entry per neuron.
    """
    modules = [m for m in network._module_list]
    # layer_infos[i] with a weight corresponds to the i-th parametrised module;
    # layer_infos[0] is the input layer, which has no module.
    n_param_before = sum(
        1 for li in network._layer_infos[:layer_idx]
        if li.connection_layer_kwargs.get("out_features") is not None
        or li.connection_layer_kwargs.get("out_channels") is not None
    )
    w_in = modules[n_param_before].weight.detach().numpy()
    w_out = modules[n_param_before + 1].weight.detach().numpy()

    in_norm = np.linalg.norm(w_in, axis=1)      # per output neuron
    out_norm = np.linalg.norm(w_out, axis=0)    # per input neuron
    return {
        "in_norm": in_norm,
        "out_norm": out_norm,
        "in_rms": in_norm / np.sqrt(w_in.shape[1]),
        "out_rms": out_norm / np.sqrt(w_out.shape[0]),
        "in_norm_rel": in_norm / in_norm.mean(),
        "out_norm_rel": out_norm / out_norm.mean(),
        "d_in": np.full(len(in_norm), w_in.shape[1]),
        "d_out": np.full(len(out_norm), w_out.shape[0]),
    }


def activation_importances(activations, active_threshold=0.5):
    """Activation statistics per neuron of one layer.

    Args:
        activations: ndarray of shape (n_samples, n_neurons).
        active_threshold: a sample counts as "active" if |a| exceeds this.
    """
    return {
        "act_std": activations.std(axis=0),
        "mean_abs": np.abs(activations).mean(axis=0),
        "act_pct": (np.abs(activations) > active_threshold).mean(axis=0),
    }


def taylor_saliency(network, x, y, layer_labels, quantizers=None):
    """First-order Taylor importance ``mean_s |dL/da * a|`` in loss units.

    Gradients are taken with identity quantizers by default: the deterministic
    quantizer rounds, whose gradient is zero almost everywhere, would make the
    saliency vanish.

    Returns:
        dict mapping layer label -> 1-D numpy array over neurons.
    """
    if quantizers is None:
        quantizers = [lambda t: t] * len(network._layer_infos)
    out, acts = network.forward(x, quantizers=quantizers,
                                return_activations_torch=True)
    for label in layer_labels:
        acts[label].retain_grad()
    F.cross_entropy(out, y).backward()
    return {
        label: (acts[label].grad * acts[label]).abs().mean(0).detach().numpy()
        * len(x)
        for label in layer_labels
    }


def collect_importances(experiment, x, y, layer_indices, run_ids, chapter_id,
                        quantizers, taylor_subset=10_000):
    """Assemble all importance measures for every neuron, layer and run.

    Returns:
        pandas.DataFrame with one row per (run, layer, neuron).
    """
    import pandas as pd

    labels = [f"L{i + 1}" for i in layer_indices]
    rows = []
    for run_id in run_ids:
        experiment.load_checkpoint(run_id, chapter_id)
        network = experiment.network
        network.eval()
        with torch.no_grad():
            _, acts = network.forward(x, quantizers=quantizers,
                                      return_activations=True)
        tay = taylor_saliency(network, x[:taylor_subset], y[:taylor_subset],
                              labels)
        for layer_idx, label in zip(layer_indices, labels):
            w = weight_importances(network, layer_idx)
            a = activation_importances(acts[label])
            n_neurons = len(w["in_norm"])
            for i in range(n_neurons):
                row = {"run_id": run_id, "layer": label, "neuron": i + 1,
                       "taylor": tay[label][i]}
                row.update({k: v[i] for k, v in w.items()})
                row.update({k: v[i] for k, v in a.items()})
                rows.append(row)
    return pd.DataFrame(rows)
