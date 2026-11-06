"""Single-neuron lesioning of a trained network.

A lesion replaces one neuron's (quantized) activation by a surrogate that
carries no information about the target, and measures the resulting drop in
test performance. Three surrogates are provided:

``zero``     clamp the neuron to 0 (knock-out).
``mean``     clamp it to its dataset-mean activation.
``shuffle``  permute its activations across the samples of the batch, which
             preserves the neuron's marginal distribution while destroying
             its dependence on the input.

All three satisfy the Markov condition ``Y -> X_-i -> X_lesioned``, so the
information available to any downstream decoder is exactly
:math:`I(\\bm X_{-i};Y)` -- see the ablation bounds in the paper appendix.

Lesions are injected by *wrapping the layer's quantizer*, so no surgery on
the network is required and the intervention lands on exactly the activations
that the information-theoretic analysis sees.
"""

import torch

__all__ = [
    "ZeroLesion",
    "MeanLesion",
    "ShuffleLesion",
    "make_lesion",
    "layer_mean_activations",
    "lesion_scan",
]


class ZeroLesion:
    """Quantize, then clamp one neuron to zero.

    With ``rescale=True`` the surviving neurons are scaled by n/(n-1)
    (inverted-dropout compensation). The rescaled values leave the
    quantization grid, which is harmless because the layer's output is not
    quantized again downstream.
    """

    def __init__(self, base_quantizer, neuron_idx, rescale=False):
        self._base = base_quantizer
        self._neuron_idx = neuron_idx
        self._rescale = rescale

    def __call__(self, x):
        x = self._base(x).clone()
        x[:, self._neuron_idx] = 0.0
        if self._rescale:
            n = x.shape[1]
            x = x * (n / (n - 1))
        return x


class MeanLesion:
    """Quantize, then clamp one neuron to a fixed (dataset-mean) value."""

    def __init__(self, base_quantizer, neuron_idx, mean_value):
        self._base = base_quantizer
        self._neuron_idx = neuron_idx
        self._mean_value = float(mean_value)

    def __call__(self, x):
        x = self._base(x).clone()
        x[:, self._neuron_idx] = self._mean_value
        return x


class ShuffleLesion:
    """Quantize, then permute one neuron's values across the batch.

    Preserves the neuron's marginal distribution exactly while removing its
    dependence on the input (permutation importance in the sense of Breiman).
    """

    def __init__(self, base_quantizer, neuron_idx, seed=0):
        self._base = base_quantizer
        self._neuron_idx = neuron_idx
        self._generator = torch.Generator().manual_seed(int(seed))

    def __call__(self, x):
        x = self._base(x).clone()
        perm = torch.randperm(x.shape[0], generator=self._generator)
        x[:, self._neuron_idx] = x[perm, self._neuron_idx]
        return x


def make_lesion(mode, base_quantizer, neuron_idx, mean_value=None,
                rescale=False, seed=0):
    """Factory dispatching on ``mode`` in {'zero', 'mean', 'shuffle'}."""
    if mode == "zero":
        return ZeroLesion(base_quantizer, neuron_idx, rescale=rescale)
    if mode == "mean":
        if mean_value is None:
            raise ValueError("mode='mean' requires mean_value")
        return MeanLesion(base_quantizer, neuron_idx, mean_value)
    if mode == "shuffle":
        return ShuffleLesion(base_quantizer, neuron_idx, seed=seed)
    raise NotImplementedError(f"Unknown lesion mode {mode!r}")


def layer_mean_activations(network, x, quantizers, layer_label,
                           batch_size=20_000):
    """Per-neuron mean of the quantized activations of one layer over x."""
    total = None
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            _, acts = network.forward(
                x[start:start + batch_size], quantizers=quantizers,
                return_activations=True,
            )
            batch_sum = acts[layer_label].sum(axis=0)
            total = batch_sum if total is None else total + batch_sum
    return total / len(x)


def _accuracy(network, x, y, quantizers, batch_size=20_000):
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out = network.forward(x[start:start + batch_size],
                                  quantizers=quantizers)
            correct += (out.argmax(dim=1)
                        == y[start:start + batch_size]).sum().item()
    return correct / len(x)


def lesion_scan(experiment, x, y, base_quantizers, layer_indices,
                run_ids, chapter_id, mode="zero", rescale=False,
                batch_size=20_000, progress=None):
    """Lesion every neuron of the given layers, for every run.

    Args:
        experiment: Experiment instance (checkpoints are loaded in place).
        x, y: evaluation inputs and integer labels (torch tensors).
        base_quantizers: quantizer list for the unlesioned network.
        layer_indices: zero-based ``layer_infos`` indices to lesion. The
            activation label of index i is ``L{i+1}``.
        run_ids: iterable of run ids.
        chapter_id: checkpoint chapter to evaluate at.
        mode: 'zero', 'mean' or 'shuffle'.
        rescale: only for mode='zero'; scale survivors by n/(n-1).
        progress: optional callable wrapping the run iterator (e.g. tqdm).

    Returns:
        pandas.DataFrame with one row per (run, layer, neuron).
    """
    import pandas as pd

    if rescale and mode != "zero":
        raise ValueError("rescale only applies to mode='zero'")

    records = []
    runs = progress(run_ids) if progress else run_ids
    for run_id in runs:
        experiment.load_checkpoint(run_id, chapter_id)
        network = experiment.network
        network.eval()
        epoch_id = experiment.schedule.get_epoch_for_chapter(chapter_id)
        baseline = _accuracy(network, x, y, base_quantizers, batch_size)

        for layer_idx in layer_indices:
            layer_label = f"L{layer_idx + 1}"
            n_neurons = network._layer_infos[layer_idx].connection_layer_kwargs[
                "out_features"
            ]
            means = (
                layer_mean_activations(network, x, base_quantizers,
                                       layer_label, batch_size)
                if mode == "mean" else None
            )

            for neuron in range(n_neurons):
                lesioned = list(base_quantizers)
                lesioned[layer_idx] = make_lesion(
                    mode, base_quantizers[layer_idx], neuron,
                    mean_value=None if means is None else means[neuron],
                    rescale=rescale,
                    seed=1000 * run_id + 10 * layer_idx + neuron,
                )
                accuracy = _accuracy(network, x, y, lesioned, batch_size)
                records.append(
                    {
                        "run_id": run_id,
                        "chapter_id": chapter_id,
                        "epoch_id": epoch_id,
                        "layer": layer_label,
                        "neuron": neuron + 1,  # 1-based, matching NeuronID
                        "baseline_accuracy": baseline,
                        "lesioned_accuracy": accuracy,
                        "accuracy_drop": baseline - accuracy,
                    }
                )
    return pd.DataFrame(records)
