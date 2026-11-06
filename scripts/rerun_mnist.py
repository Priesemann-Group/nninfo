"""Add training runs to an existing experiment via Experiment.rerun().

Replays run 0's checkpoint schedule with fresh random weight
initializations, appending new run ids to the same experiment directory
(the nninfo-native way to build statistics across runs).

One-neuron dropout is a `one_neuron_dropout` layer of the network (see
nninfo.model.neural_network.OneNeuronDropout), so it is part of the network
config that `Experiment.load()` reconstructs, and reruns inherit the
intervention automatically -- nothing special is needed here. The script
prints the dropout layers it restored, so a run that silently lost its
intervention cannot go unnoticed.

Concurrency note: rerun() claims the next run id by scanning the checkpoint
directory at startup. Parallel rerun processes on the same experiment must
therefore be launched staggered -- start the next one only after the previous
has written its first checkpoint (ckpt_r<run>_c000000_*) -- or given explicit
`--run-id`s.

Run from the repository root:

    python scripts/rerun_mnist.py --experiment-id mnist_8levels_onehot_4x10
"""

import argparse

import nninfo
from nninfo.model.neural_network import OneNeuronDropout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of additional runs to train (default 1).")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Train exactly one new run under this explicit "
                        "run id, replaying the experiment's saved schedule "
                        "(Experiment.rerun_original_schedule). Unlike the "
                        "default mode this works while run 0 is still "
                        "training and is safe to launch in parallel; "
                        "--n-runs is ignored.")
    args = parser.parse_args()

    experiment = nninfo.Experiment.load(args.experiment_id)

    dropout_layers = [
        i for i, layer in enumerate(experiment.network._connection_layers)
        if isinstance(layer, OneNeuronDropout)
    ]
    if dropout_layers:
        ps = [experiment.network._connection_layers[i].p for i in dropout_layers]
        print(f"one-neuron dropout layers at layer_info indices "
              f"{dropout_layers} (labels {[f'L{i+1}' for i in dropout_layers]}, "
              f"p={ps})", flush=True)

    if args.run_id is not None:
        experiment.rerun_original_schedule(run_id=args.run_id)
    else:
        experiment.rerun(args.n_runs)

    print(f"[{args.experiment_id}] rerun complete; runs on disk: "
          f"{experiment.checkpoint_manager.get_run_ids()}", flush=True)


if __name__ == "__main__":
    main()
