"""Tests for the MNIST training script (scripts/train_mnist.py)."""

import importlib.util
from pathlib import Path

import pytest
import torch

import nninfo
from nninfo.model.neural_network import OneNeuronDropout
from nninfo.model.quantization import Q3Quantizer


def _load_script():
    path = Path(__file__).parent.parent / "scripts" / "train_mnist.py"
    spec = importlib.util.spec_from_file_location("train_mnist", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


class TestDropoutLayerInsertion:
    """Dropout is a layer of the network, inserted after each targeted layer."""

    def test_inserts_one_layer_after_each_target(self):
        infos = script.make_layer_infos((5, 5, 5), dropout_layers=(3, 4, 5))
        kinds = [li.connection_layer for li in infos]
        assert kinds == ["input", "linear", "linear", "linear",
                         "one_neuron_dropout", "linear", "one_neuron_dropout",
                         "linear", "one_neuron_dropout", "linear"]

    def test_records_the_probability(self):
        infos = script.make_layer_infos((5, 5, 5), dropout_layers=(4,),
                                        dropout_p=0.5)
        assert infos[5].connection_layer == "one_neuron_dropout"
        assert infos[5].connection_layer_kwargs == {"p": 0.5}
        assert infos[5].activation_function is None

    def test_single_target_leaves_the_rest_intact(self):
        plain = script.make_layer_infos((5, 5, 5))
        infos = script.make_layer_infos((5, 5, 5), dropout_layers=(4,))
        assert infos[:5] == plain[:5]
        assert infos[6:] == plain[5:]

    def test_quantizer_gains_a_none_per_inserted_layer(self):
        q = script.train_quantizer(8, n_hidden=5, dropout_layers=(3, 4, 5))
        infos = script.make_layer_infos((5, 5, 5), dropout_layers=(3, 4, 5))
        assert len(q) == len(infos)  # the list is indexed by layer position
        assert [i for i, p in enumerate(q) if p is None] == [0, 4, 6, 8, 9]

    def test_quantized_layers_keep_their_levels(self):
        q = script.train_quantizer(8, n_hidden=5, dropout_layers=(3, 4, 5))
        assert [p["levels"] for p in q if p is not None] == [8, 8, 8, 8, 8]

    def test_dropout_on_unquantized_layer_rejected(self):
        with pytest.raises(ValueError):
            script.train_quantizer(8, n_hidden=5, dropout_layers=(0,))

    def test_label_map_for_all_three_bottlenecks(self):
        # each insertion shifts everything after it by one
        assert script.dropout_label_map((3, 4, 5)) == {4: (4, 5), 5: (6, 7),
                                                       6: (8, 9)}

    def test_label_map_for_a_single_layer(self):
        assert script.dropout_label_map((4,)) == {5: (5, 6)}

    def test_network_builds_and_round_trips(self):
        infos = script.make_layer_infos((5, 5, 5), dropout_layers=(3,),
                                        dropout_p=0.5)
        net = nninfo.NeuralNetwork(layer_infos=infos, init_str="xavier")
        restored = nninfo.NeuralNetwork.from_config(net.to_config())
        layer = restored._connection_layers[4]
        assert isinstance(layer, OneNeuronDropout) and layer.p == 0.5

    def test_measurement_quantizer_skips_dropout_layers(self):
        infos = script.make_layer_infos((5, 5, 5), dropout_layers=(3, 4, 5))
        net = nninfo.NeuralNetwork(layer_infos=infos, init_str="xavier")
        q = script.measurement_quantizer_for_network(net, levels=8)
        assert len(q) == len(infos)
        assert [i for i, p in enumerate(q) if p is None] == [0, 4, 6, 8, 9]
        assert all(p["rounding_point"] == "center_saturating"
                   for p in q if p is not None)

    def test_measurement_quantizer_respects_graded_levels(self):
        infos = script.make_layer_infos((5, 5, 5), dropout_layers=(3,))
        net = nninfo.NeuralNetwork(layer_infos=infos, init_str="xavier")
        q = script.measurement_quantizer_for_network(net, levels=8,
                                                    bottleneck_levels=[16, 8, 4])
        assert [p["levels"] for p in q if p is not None] == [8, 8, 16, 8, 4]


class TestSchedule:
    def test_matches_original_log_spaced_grid(self):
        schedule = script.make_schedule(10_000)
        ends = schedule.chapter_ends
        assert ends[-1] == 10_000
        # same grid as the original experiment (10 chapters per decade)
        assert ends[:14] == [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32]
        assert all(isinstance(e, int) for e in ends)
        assert ends == sorted(set(ends))  # strictly increasing

    def test_short_run(self):
        schedule = script.make_schedule(2)
        assert schedule.chapter_ends[0] == 0
        assert schedule.chapter_ends[-1] == 2


class TestArchitecture:
    def test_matches_original_experiment(self):
        infos = script.make_layer_infos()
        assert len(infos) == 7
        dims = [
            (li.connection_layer_kwargs.get("in_features"),
             li.connection_layer_kwargs.get("out_features"))
            for li in infos[1:]
        ]
        assert dims == [(784, 50), (50, 10), (10, 5), (5, 5), (5, 5), (5, 10)]
        assert infos[-1].activation_function == "softmax_output"
        assert all(li.activation_function == "tanh" for li in infos[1:-1])
        # dropout targets are exactly the five-neuron layers
        five = [i for i, li in enumerate(infos)
                if li.connection_layer_kwargs.get("out_features") == 5]
        assert five == list(script.DROPOUT_LAYER_INDICES)

    def test_extra_five_neuron_layer(self):
        infos = script.make_layer_infos((5, 5, 5, 5))
        dims = [
            (li.connection_layer_kwargs.get("in_features"),
             li.connection_layer_kwargs.get("out_features"))
            for li in infos[1:]
        ]
        assert dims == [(784, 50), (50, 10), (10, 5), (5, 5), (5, 5), (5, 5),
                        (5, 10)]
        assert infos[-1].activation_function == "softmax_output"

    def test_ten_neuron_bottlenecks(self):
        infos = script.make_layer_infos((10, 10, 10))
        dims = [
            (li.connection_layer_kwargs.get("in_features"),
             li.connection_layer_kwargs.get("out_features"))
            for li in infos[1:]
        ]
        assert dims == [(784, 50), (50, 10), (10, 10), (10, 10), (10, 10),
                        (10, 10)]

    def test_quantizer_length_follows_architecture(self):
        q = script.train_quantizer(8, n_hidden=6)
        assert len(q) == 8
        assert q[0] is None and q[-1] is None
        assert all(e == {"levels": 8, "rounding_point": "stochastic"}
                   for e in q[1:-1])

    def test_dropout_indices_default_architecture(self):
        infos = script.make_layer_infos((5, 5, 5))
        assert script.dropout_layer_indices(infos, 5) == (3, 4, 5)

    def test_dropout_indices_4x10_architecture(self):
        # all four ten-neuron layers (incl. the 50->10 layer); the ten-neuron
        # softmax output layer is excluded
        infos = script.make_layer_infos((10, 10, 10))
        assert script.dropout_layer_indices(infos, 10) == (2, 3, 4, 5)

    def test_graded_bottleneck_levels(self):
        # 3x8 net: 5 hidden layers, graded levels on the last three
        q = script.train_quantizer(8, n_hidden=5, bottleneck_levels=[16, 8, 4])
        assert [e["levels"] for e in q[1:-1]] == [8, 8, 16, 8, 4]
        assert q[0] is None and q[-1] is None
        assert all(e["rounding_point"] == "stochastic" for e in q[1:-1])

    def test_graded_levels_reverse_control(self):
        q = script.train_quantizer(8, n_hidden=5, bottleneck_levels=[4, 8, 16])
        assert [e["levels"] for e in q[1:-1]] == [8, 8, 4, 8, 16]

    def test_measurement_quantizer_matches_graded_layout(self):
        q = script.measurement_quantizer(8, n_hidden=5, bottleneck_levels=[16, 8, 4])
        assert [e["levels"] for e in q[1:-1]] == [8, 8, 16, 8, 4]
        assert all(e["rounding_point"] == "center_saturating" for e in q[1:-1])

    def test_uniform_levels_unchanged_by_new_arg(self):
        assert script.train_quantizer(8, 5) == script.train_quantizer(8, 5, None)

    def test_dropout_and_graded_levels_compose(self):
        q = script.train_quantizer(8, n_hidden=5, bottleneck_levels=[16, 8, 4],
                                   dropout_layers=(3, 5))
        # levels unchanged, with a None inserted after each targeted layer
        assert [e["levels"] for e in q if e is not None] == [8, 8, 16, 8, 4]
        assert [i for i, e in enumerate(q) if e is None] == [0, 4, 7, 8]
