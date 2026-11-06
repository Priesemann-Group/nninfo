"""Single-neuron lesioning of a trained MNIST classifier.

Thin CLI around :mod:`nninfo.analysis.lesioning`. For each five-neuron layer
of an experiment, every neuron is lesioned in turn and the resulting drop in
test accuracy is recorded, averaged over neurons within a layer and
aggregated over runs.

Hypothesis: layers whose encodings are more vulnerable in the
Shannon-observable sense should show a larger accuracy drop.

Layer labelling: nninfo labels every entry of ``layer_infos`` including the
input layer, so the five-neuron layers of the 784-50-10-5-5-5-10 network are
L4, L5, L6 (the papers call the same layers L3-L5).

Evaluation uses the experiment's own training quantizer (stochastic rounding,
matching the deg_red / deg_vuln measurements), so networks with
one_neuron_dropout layers work too. Reproducible despite the stochastic
rounding: load_checkpoint() restores the saved torch RNG state before every
evaluation.

Run from the repository root, e.g.:

    python scripts/lesioning_mnist_8levels_onehot.py
    python scripts/lesioning_mnist_8levels_onehot.py --mode shuffle
    python scripts/lesioning_mnist_8levels_onehot.py \
        --experiment-id mnist_8levels_onehot_dropout_L4 --layers 4
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import nninfo
from nninfo.analysis.lesioning import lesion_scan
from nninfo.model.quantization import quantizer_list_factory

EXPERIMENT_ID = "mnist_8levels_onehot"
EVAL_BATCH_SIZE = 20_000
PAPER_EPOCH = 10_000  # the paper lesions the fully trained networks at epoch 10^4


def five_neuron_layer_indices(network):
    """Zero-based layer_infos indices of all layers with 5 output neurons."""
    return [
        i
        for i, info in enumerate(network._layer_infos)
        if info.connection_layer_kwargs.get("out_features") == 5
    ]


def load_test_data(experiment):
    """Test set (x, y) from the pipeline, with a label sanity check."""
    dataset = experiment.task["full_set/test"]
    x, y = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    y = y.long()
    assert y.min() >= 0 and y.max() <= 9 and len(torch.unique(y)) == 10, (
        "Test labels look invalid (QMNIST parsing regression?)"
    )
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument(
        "--runs", type=int, default=None,
        help="Number of runs to analyze (default: all runs with checkpoints).",
    )
    parser.add_argument(
        "--layers", type=int, nargs="*", default=None,
        help="Layer labels to lesion as L-numbers, e.g. --layers 4 5 6 "
        "(default: all five-neuron layers).",
    )
    parser.add_argument(
        "--rescale", action="store_true",
        help="Rescale surviving activations by n/(n-1) after the lesion "
        "(inverted-dropout compensation). Only valid with --mode zero.",
    )
    parser.add_argument(
        "--chapter", type=int, default=None,
        help="Chapter to lesion at (default: the chapter at epoch 10^4, as in "
        "the paper). Outputs get an '_e<epoch>' suffix.",
    )
    parser.add_argument(
        "--mode", choices=["zero", "shuffle", "mean"], default="zero",
        help="Lesion surrogate: clamp to zero (default), permute across "
        "samples, or clamp to the dataset mean.",
    )
    args = parser.parse_args()
    suffix = {"zero": "_rescaled" if args.rescale else "",
              "shuffle": "_shuffle", "mean": "_mean"}[args.mode]

    experiment = nninfo.Experiment.load(args.experiment_id)

    run_ids = experiment.checkpoint_manager.get_run_ids()
    if args.runs is not None:
        run_ids = run_ids[: args.runs]

    layer_indices = five_neuron_layer_indices(experiment.network)
    if args.layers:
        requested = [l - 1 for l in args.layers]
        assert set(requested) <= set(layer_indices), (
            f"Requested layers {args.layers} are not all five-neuron layers "
            f"(five-neuron layers are {[i + 1 for i in layer_indices]})"
        )
        layer_indices = requested
    layer_labels = [f"L{i + 1}" for i in layer_indices]
    print(f"Lesioning layers {layer_labels} on runs {run_ids[0]}..{run_ids[-1]}")

    x_test, y_test = load_test_data(experiment)
    print(f"Test set: {len(x_test)} samples")

    base_quantizers = quantizer_list_factory(
        experiment.trainer.to_config()["quantizer"],
        experiment.network.get_limits_list(),
    )

    chapter = args.chapter
    if chapter is None:
        chapter = experiment.schedule.chapter_ends.index(PAPER_EPOCH)
    suffix += f"_e{experiment.schedule.get_epoch_for_chapter(chapter)}"

    frames = []
    for run_id in tqdm(run_ids, desc="runs"):
        frames.append(
            lesion_scan(
                experiment, x_test, y_test, base_quantizers,
                layer_indices=layer_indices, run_ids=[run_id],
                chapter_id=chapter, mode=args.mode,
                rescale=args.rescale, batch_size=EVAL_BATCH_SIZE,
            )
        )
    import pandas as pd

    results = pd.concat(frames, ignore_index=True)

    out_dir = experiment.experiment_dir / "lesioning"
    out_dir.mkdir(exist_ok=True)
    results.to_csv(out_dir / f"results{suffix}.csv", index=False)
    print(f"Saved results to {out_dir / f'results{suffix}.csv'}")
    print(f"Mean baseline accuracy: {results.baseline_accuracy.mean():.4f}")

    plot(results, layer_labels, experiment.experiment_dir / "plots",
         suffix=suffix)


def plot(results, layer_labels, plot_dir, suffix=""):
    cm = 1 / 2.54

    # neuron-averaged drop per run -> distribution over runs per layer
    per_run = (
        results.groupby(["layer", "run_id"])["accuracy_drop"].mean().reset_index()
    )

    fig, ax = plt.subplots(figsize=(8 * cm, 7 * cm), dpi=150)

    means = [per_run[per_run.layer == l]["accuracy_drop"].mean() for l in layer_labels]
    sems = [per_run[per_run.layer == l]["accuracy_drop"].sem() for l in layer_labels]
    xs = np.arange(len(layer_labels))

    # the sem is undefined for single-run experiments
    ax.bar(xs, means, yerr=np.nan_to_num(sems), capsize=3, color="lightsteelblue",
           edgecolor="k", linewidth=0.8, zorder=2)

    # per-run points (neuron-averaged), jittered
    rng = np.random.default_rng(0)
    for x, label in zip(xs, layer_labels):
        vals = per_run[per_run.layer == label]["accuracy_drop"]
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(x + jitter, vals, s=4, c="k", alpha=0.4, zorder=3, lw=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(layer_labels)
    ax.set_xlabel("Lesioned layer")
    ax.set_ylabel("Test accuracy drop", fontsize=9)
    epoch = int(results.epoch_id.iloc[0])
    variant = ""
    if "rescaled" in suffix:
        variant = "; survivors rescaled 5/4"
    elif "shuffle" in suffix:
        variant = "; shuffle lesion"
    elif "mean" in suffix:
        variant = "; mean lesion"
    ax.set_title(
        f"Single-neuron lesioning at epoch {epoch}\n"
        f"(avg. over neurons; dots: individual runs{variant})",
        fontsize=8,
    )
    ax.spines[["top", "right"]].set_visible(False)

    plot_dir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(plot_dir / f"lesioning{suffix}.{ext}", bbox_inches="tight")
    print(f"Saved plot to {plot_dir / f'lesioning{suffix}.pdf'}")


if __name__ == "__main__":
    main()
