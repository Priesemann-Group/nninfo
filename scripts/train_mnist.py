"""Train an MNIST run with the settings of exp_mnist_8levels_onehot.

Replicates the original experiment (same architecture, task, optimizer,
quantization and checkpoint style), with two optional variations:

--levels N   number of quantization levels for the hidden layers (default 8)
--dropout    one-neuron dropout: during training, zero out one randomly
             chosen neuron per sample in each of the three five-neuron hidden
             layers. Activations are quantized as usual (stochastic rounding),
             then one neuron per sample is clamped to zero, without rescaling
             the survivors.

Dropout is a `one_neuron_dropout` *layer* inserted after each targeted layer,
so it is part of the network config and round-trips through
`Experiment.load()` / `Experiment.rerun()`. Being a layer of its own means the
activations before and after the intervention are separately labelled and can
be measured separately; like `nn.Dropout` it is an identity under `eval()`, so
the testing and measurement paths see the intact code.

Layer labelling: `--dropout-layers` takes labels of the *dropout-free* net
(4 5 6 are the three five-neuron layers). Each inserted layer shifts the
labels after it, so with dropout on L4, L5 and L6 the five-neuron layers end
up at L4, L6 and L8, with their post-dropout copies at L5, L7 and L9. The
script prints the resulting map.

Training length is given in epochs. Checkpoints follow the original
experiment's log-spaced chapter grid (10 chapters per decade:
0, 1, 2, ..., 8, 10, 13, ..., n_epochs).

Run from the repository root:

    python scripts/train_mnist.py --experiment-id my_run           # original settings
    python scripts/train_mnist.py --experiment-id my_run6 --levels 6
    python scripts/train_mnist.py --experiment-id my_dropout --dropout
"""

import argparse

import numpy as np
import torch

import nninfo
import nninfo.utils
from nninfo.model.neural_network import OneNeuronDropout
from nninfo.model.quantization import quantizer_list_factory

DEFAULT_EPOCHS = 10_000
CHAPTERS_PER_DECADE = 10  # log-spaced checkpoint grid, as in the original run

# layer_info indices of the three five-neuron layers (labels L4, L5, L6) in
# the default architecture; other architectures compute their own indices via
# dropout_layer_indices()
DROPOUT_LAYER_INDICES = (3, 4, 5)


def dropout_layer_indices(layer_infos, width):
    """Layer_info indices of all hidden (tanh) layers with `width` neurons.

    The softmax output layer is excluded even if it has the same width."""
    return tuple(
        i for i, li in enumerate(layer_infos)
        if li.activation_function == "tanh"
        and li.connection_layer_kwargs.get("out_features") == width
    )

BATCH_SIZE = 64

def _quantizer(levels, n_hidden, rounding, bottleneck_levels=None,
               dropout_layers=()):
    """Quantizer params for the n_hidden quantized layers (input/output are None).

    `levels` applies to every hidden layer; `bottleneck_levels`, if given,
    overrides the trailing layers one value per bottleneck, enabling a graded
    capacity gradient at constant layer width (e.g. 16/8/4).

    `dropout_layers` holds the layer_infos indices (of the dropout-free net)
    that a `one_neuron_dropout` layer is inserted after. Each insertion adds a
    `None` entry, since the dropout layer itself is not quantized -- the
    quantizer list is indexed by layer_infos position and must stay aligned.
    """
    per_layer = n_hidden * [levels]
    if bottleneck_levels is not None:
        assert len(bottleneck_levels) <= n_hidden
        per_layer[n_hidden - len(bottleneck_levels):] = list(bottleneck_levels)
    params = ([None]
              + [{"levels": lv, "rounding_point": rounding} for lv in per_layer]
              + [None])
    for i in sorted(dropout_layers, reverse=True):
        if params[i] is None:
            raise ValueError(f"cannot apply dropout to unquantized layer index {i}")
        params.insert(i + 1, None)
    return params


def train_quantizer(levels=8, n_hidden=5, bottleneck_levels=None,
                    dropout_layers=()):
    return _quantizer(levels, n_hidden, "stochastic", bottleneck_levels,
                      dropout_layers)


def measurement_quantizer(levels=8, n_hidden=5, bottleneck_levels=None,
                          dropout_layers=()):
    return _quantizer(levels, n_hidden, "center_saturating", bottleneck_levels,
                      dropout_layers)


TRAIN_QUANTIZER = train_quantizer()
MEASUREMENT_QUANTIZER = measurement_quantizer()


def measurement_quantizer_for_network(network, levels=8, bottleneck_levels=None):
    """Deterministic measurement quantizer derived from an actual network.

    Reads the layer list instead of assuming a layout, so it stays correct when
    `one_neuron_dropout` layers are present (they get `None`, like the input
    and output layers).
    """
    infos = network._layer_infos
    hidden = [i for i, li in enumerate(infos) if li.activation_function == "tanh"]
    per_layer = {i: levels for i in hidden}
    if bottleneck_levels is not None:
        for i, lv in zip(hidden[-len(bottleneck_levels):], bottleneck_levels):
            per_layer[i] = lv
    return [{"levels": per_layer[i], "rounding_point": "center_saturating"}
            if i in per_layer else None
            for i in range(len(infos))]


def make_layer_infos(bottlenecks=(5, 5, 5), dropout_layers=(), dropout_p=1.0):
    """Architecture 784-50-10-<bottlenecks>-10.

    The default (5, 5, 5) is the original exp_mnist_8levels_onehot net
    (784-50-10-5-5-5-10). Passing e.g. (5, 5, 5, 5) inserts an extra
    five-neuron layer; (10, 10, 10) widens the bottlenecks to ten neurons.

    `dropout_layers` holds layer_infos indices of the dropout-free net; a
    `one_neuron_dropout` layer is inserted directly after each of them, which
    shifts the labels of all following layers (see the module docstring).
    """
    hidden_sizes = [784, 50, 10, *bottlenecks]
    layer_infos = [
        nninfo.LayerInfo(connection_layer="input", activation_function="input")
    ]
    layer_infos += [
        nninfo.LayerInfo(
            connection_layer="linear",
            connection_layer_kwargs={"in_features": n_in, "out_features": n_out},
            activation_function="tanh",
        )
        for n_in, n_out in zip(hidden_sizes[:-1], hidden_sizes[1:])
    ]
    layer_infos.append(
        nninfo.LayerInfo(
            connection_layer="linear",
            connection_layer_kwargs={"in_features": bottlenecks[-1], "out_features": 10},
            activation_function="softmax_output",
        )
    )
    for i in sorted(dropout_layers, reverse=True):
        layer_infos.insert(i + 1, nninfo.LayerInfo(
            connection_layer="one_neuron_dropout",
            connection_layer_kwargs={"p": float(dropout_p)},
            activation_function=None,
        ))
    return layer_infos


def dropout_label_map(dropout_layers):
    """{label in the dropout-free net -> (label of the layer, label after dropout)}.

    Each inserted layer shifts everything after it by one, so the k-th targeted
    layer (in ascending order) moves by the number of insertions before it.
    """
    shifted = {}
    for k, i in enumerate(sorted(dropout_layers)):
        label = i + 1 + k          # its own label after k earlier insertions
        shifted[i + 1] = (label, label + 1)
    return shifted


def make_schedule(n_epochs):
    """Log-spaced chapter grid matching the original experiment
    (10 chapters per decade: 0, 1, 2, ..., 8, 10, 13, ..., n_epochs)."""
    n_chapters = max(1, int(round(CHAPTERS_PER_DECADE * np.log10(n_epochs))))
    schedule = nninfo.Schedule()
    schedule.create_log_spaced_chapters(n_epochs, n_chapters)
    return schedule


def build_experiment(experiment_id, n_epochs, dropout=False, levels=8,
                     bottlenecks=(5, 5, 5), bottleneck_levels=None,
                     dropout_layer_labels=None, dropout_p=1.0):
    n_hidden = len(bottlenecks) + 2

    if bottleneck_levels is not None:
        assert len(bottleneck_levels) == len(bottlenecks), (
            "--bottleneck-levels needs one value per bottleneck layer"
        )

    task = nninfo.TaskManager("mnist_1d_dat")
    task["full_set"].train_test_val_sequential_split(60_000, 60_000, 0)

    # indices refer to the dropout-free architecture; dropout layers are then
    # inserted after them, which shifts the labels of all following layers
    plain_layer_infos = make_layer_infos(bottlenecks)
    dropout_indices = ()
    if dropout:
        if dropout_layer_labels:
            # explicit L-labels: label n corresponds to layer_infos[n-1]
            dropout_indices = tuple(l - 1 for l in dropout_layer_labels)
            for i in dropout_indices:
                assert plain_layer_infos[i].activation_function == "tanh", (
                    f"L{i+1} is not a hidden tanh layer"
                )
        else:
            # default: all hidden layers with the bottleneck width, e.g.
            # (3, 4, 5) for 5-5-5 and (2, 3, 4, 5) for 10-10-10
            dropout_indices = dropout_layer_indices(plain_layer_infos,
                                                    bottlenecks[-1])

    layer_infos = make_layer_infos(bottlenecks, dropout_indices, dropout_p)
    network = nninfo.NeuralNetwork(layer_infos=layer_infos, init_str="xavier")

    trainer = nninfo.Trainer(
        dataset_name="full_set/train",
        optim_str="SGD",
        loss_str="CELoss",
        lr=0.01,
        shuffle=True,
        batch_size=BATCH_SIZE,
        quantizer=train_quantizer(levels, n_hidden, bottleneck_levels,
                                  dropout_layers=dropout_indices),
    )
    tester = nninfo.Tester(dataset_name="full_set/test")
    schedule = make_schedule(n_epochs)

    return nninfo.Experiment(experiment_id, network, task, trainer, tester, schedule)


def final_test_accuracy(experiment, levels=8, bottleneck_levels=None):
    """Accuracy of the current network state, deterministic quantization,
    no dropout. Labels come from the pipeline (nninfo's qmnist_class_labels
    guards against the torchvision QMNIST endianness bug)."""
    from torch.utils.data import DataLoader

    dataset = experiment.task["full_set/test"]
    x, y = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    y = y.long()
    assert y.min() >= 0 and y.max() <= 9 and len(torch.unique(y)) == 10, (
        "Test labels look invalid (QMNIST parsing regression?)"
    )

    quantizers = quantizer_list_factory(
        measurement_quantizer_for_network(experiment.network, levels,
                                         bottleneck_levels),
        experiment.network.get_limits_list(),
    )
    experiment.network.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), 20_000):
            out = experiment.network.forward(
                x[start : start + 20_000], quantizers=quantizers
            )
            correct += (out.argmax(dim=1) == y[start : start + 20_000]).sum().item()
    return correct / len(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Number of epochs to train (default 10000).")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove an existing experiment with this id first.")
    parser.add_argument("--dropout", action="store_true",
                        help="Enable one-neuron dropout on the five-neuron "
                        "layers during training (default: off, i.e. original "
                        "settings).")
    parser.add_argument("--levels", type=int, default=8,
                        help="Number of quantization levels for the hidden "
                        "layers (default 8, as in the original run).")
    parser.add_argument("--bottlenecks", type=int, nargs="+", default=[5, 5, 5],
                        help="Widths of the layers after 784-50-10 (default "
                        "5 5 5, as in the original run).")
    parser.add_argument("--bottleneck-levels", type=int, nargs="+", default=None,
                        help="Per-bottleneck quantization levels, one value "
                        "per bottleneck layer (e.g. 16 8 4). Overrides "
                        "--levels for those layers, giving a graded capacity "
                        "gradient at constant layer width.")
    parser.add_argument("--dropout-p", type=float, default=1.0,
                        help="With --dropout: probability that a given sample "
                        "loses one neuron in each targeted layer (default 1.0 "
                        "= always; 0.5 = partial dropout).")
    parser.add_argument("--dropout-layers", type=int, nargs="+", default=None,
                        help="With --dropout: L-labels of the layers to apply "
                        "one-neuron dropout to (e.g. 4 for L4 only, or 4 5 6 "
                        "for all three bottlenecks). Default: all layers of "
                        "the bottleneck width.")
    args = parser.parse_args()

    if args.overwrite:
        nninfo.utils.remove_experiment(args.experiment_id, yes=True, silent=True)

    experiment = build_experiment(args.experiment_id, args.epochs,
                                  dropout=args.dropout,
                                  levels=args.levels,
                                  bottlenecks=tuple(args.bottlenecks),
                                  bottleneck_levels=args.bottleneck_levels,
                                  dropout_layer_labels=args.dropout_layers,
                                  dropout_p=args.dropout_p)
    print(f"Schedule (epochs): {experiment.schedule.chapter_ends}")

    experiment.run_following_schedule(compute_test_loss=False)

    # sanity check: dropout layers present in the network iff requested, and
    # recorded in the saved config so that a reload reconstructs them
    built = [i for i, layer in enumerate(experiment.network._connection_layers)
             if isinstance(layer, OneNeuronDropout)]
    from_config = [i for i, layer in enumerate(experiment.network.to_config()["layers"])
                   if layer["connection_layer"] == "one_neuron_dropout"]
    assert built == from_config, (
        f"Dropout layers at {built}, but config says {from_config}"
    )
    print(f"One-neuron dropout layers at layer_info indices {built} "
          f"(labels {[f'L{i+1}' for i in built]})")
    if args.dropout:
        # undo the shift each earlier insertion caused to recover the indices
        # of the targeted layers in the dropout-free architecture
        targeted = [b - 1 - k for k, b in enumerate(sorted(built))]
        label_map = dropout_label_map(targeted)
        print("  dropout-free label -> (label, post-dropout label): "
              + ", ".join(f"L{i+1} -> L{own}/L{post}"
                          for i, (own, post) in
                          ((i, label_map[i + 1]) for i in targeted)))

    last_chapter = experiment.checkpoint_manager.get_last_chapter_in_run(
        experiment.run_id
    )
    experiment.load_checkpoint(experiment.run_id, last_chapter)

    accuracy = final_test_accuracy(experiment, levels=args.levels,
                                   bottleneck_levels=args.bottleneck_levels)
    print(f"Test accuracy at final checkpoint (chapter {last_chapter}, "
          f"epoch {args.epochs}; deterministic {args.levels}-level quantizer, "
          f"no dropout): {accuracy:.4f}")


if __name__ == "__main__":
    main()
