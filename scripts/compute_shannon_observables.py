"""Compute Shannon-observable summary structures for an MNIST experiment.

Replicates the original exp_mnist_8levels_onehot measurements
(summary_measurement_L{3,4,5,6}_stochastic, as used by
shannon_observables_scripts/1_shannon_observables_mnist.ipynb): a `deg_red`
PIDMeasurement per layer with single-neuron sources and the class label Y as
target, stochastic quantization (matching training) and no additional
binning.

Layer labels follow capture_activations' convention (L1 = input copy):
L3 is the 10-neuron layer (10 sources), L4/L5/L6 are the three five-neuron
bottleneck layers (5 sources each).

Run from the repository root, e.g.:

    python scripts/compute_shannon_observables.py \
        --experiment-id mnist_6levels_onehot
"""

import argparse
import time

import yaml

import nninfo
from nninfo.model.neural_network import NeuronID


def layer_size(experiment, layer):
    """Neurons in the layer labeled L{layer} (label n = layer_infos[n-1])."""
    infos = experiment.network._layer_infos
    i = layer - 1
    while "out_features" not in infos[i].connection_layer_kwargs:
        # e.g. a one_neuron_dropout layer, which keeps its input's width
        assert i > 0, f"cannot determine the width of L{layer}"
        i -= 1
    return infos[i].connection_layer_kwargs["out_features"]


def default_layers(experiment):
    """Every hidden layer after the 50-neuron layer, i.e. L3..L{n-1}.

    `one_neuron_dropout` layers are skipped: measurements run in eval mode,
    where the intervention is an identity, so a post-dropout layer would
    reproduce the layer before it exactly. Pass `--layers` to measure one
    anyway.
    """
    infos = experiment.network._layer_infos
    return [i + 1 for i in range(2, len(infos) - 1)
            if infos[i].connection_layer != "one_neuron_dropout"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--levels", type=int, default=None,
                        help="Quantization levels to measure with, applied to "
                        "every hidden layer. Omit (recommended) to reuse the "
                        "run's own training quantizer from its config, which "
                        "is required for runs trained with per-layer "
                        "(graded) levels.")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layer labels to measure (default: every hidden "
                        "layer after the 50-neuron layer, i.e. L3..L{n-1}, "
                        "excluding one_neuron_dropout layers, which are an "
                        "identity in eval mode).")
    args = parser.parse_args()

    experiment = nninfo.Experiment.load(args.experiment_id)

    layers = args.layers or default_layers(experiment)

    if args.levels is None:
        # the trainer's stochastic quantizer, exactly as used during training
        with open(experiment.experiment_dir / "config.yaml") as f:
            quantizer_params = yaml.safe_load(f)["trainer"]["quantizer"]
        assert quantizer_params is not None, \
            "Experiment config has no trainer quantizer; pass --levels."
    else:
        # one entry per layer_infos position; one_neuron_dropout layers are
        # not quantized
        quantizer_params = (
            [None]
            + [None if info.connection_layer == "one_neuron_dropout"
               else {"levels": args.levels, "rounding_point": "stochastic"}
               for info in experiment.network._layer_infos[1:-1]]
            + [None]
        )
    print(f"measurement quantizer: "
          f"{[q['levels'] if q else None for q in quantizer_params]}", flush=True)

    for layer in layers:
        n_neurons = layer_size(experiment, layer)
        target = [NeuronID("Y", tuple())]
        sources = [[NeuronID(f"L{layer}", (i,))] for i in range(1, n_neurons + 1)]

        measurement = nninfo.analysis.PIDMeasurement(
            experiment=experiment,
            measurement_id=f"summary_measurement_L{layer}_stochastic",
            dataset_name="full_set/train",
            quantizer_params=quantizer_params,
            pid_definition="deg_red",
            pid_kwargs={"entropy_only": False},
            binning_kwargs={"binning_method": "none"},
            target_id_list=target,
            source_id_lists=sources,
        )

        tic = time.time()
        measurement.perform_measurements(run_ids="all", chapter_ids="all",
                                         exists_ok=True)
        print(f"[{args.experiment_id}] L{layer} done in "
              f"{time.time() - tic:.0f}s", flush=True)

    print(f"[{args.experiment_id}] ALL LAYERS DONE", flush=True)


if __name__ == "__main__":
    main()
