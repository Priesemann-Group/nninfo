"""Tests for NeuralNetwork (nninfo/model/neural_network.py)."""

from types import SimpleNamespace

import pytest
import torch

from nninfo.model.neural_network import (
    ACTIV_FUNCS_BINNING_LIMITS,
    LayerInfo,
    NeuralNetwork,
    NoisyNeuralNetwork,
)


def make_layer_infos():
    return [
        LayerInfo(connection_layer="input", activation_function="input"),
        LayerInfo(
            connection_layer="linear",
            connection_layer_kwargs={"in_features": 4, "out_features": 3},
            activation_function="tanh",
        ),
        LayerInfo(
            connection_layer="linear",
            connection_layer_kwargs={"in_features": 3, "out_features": 2},
            activation_function="sigmoid",
        ),
    ]


def make_network():
    return NeuralNetwork(layer_infos=make_layer_infos(), init_str="xavier")


def identity_quantizers(n):
    return [lambda x: x] * n


class TestForward:
    def test_output_shape(self):
        net = make_network()
        x = torch.rand(5, 4)
        out = net.forward(x, quantizers=identity_quantizers(3))
        assert out.shape == (5, 2)

    def test_output_ranges_match_activations(self):
        net = make_network()
        x = torch.rand(7, 4)
        _, acts = net.forward(
            x, quantizers=identity_quantizers(3), return_activations=True
        )
        assert set(acts.keys()) == {"L1", "L2", "L3"}
        assert acts["L2"].min() >= -1.0 and acts["L2"].max() <= 1.0  # tanh
        assert acts["L3"].min() >= 0.0 and acts["L3"].max() <= 1.0  # sigmoid

    def test_forward_deterministic(self):
        net = make_network()
        x = torch.rand(5, 4)
        q = identity_quantizers(3)
        assert torch.equal(net.forward(x, q), net.forward(x, q))


class TestExtractActivations:
    def test_keys_and_types(self):
        net = make_network()
        x = torch.rand(6, 4)
        acts = net.extract_activations(x, quantizer_params=None)
        assert set(acts.keys()) == {"L1", "L2", "L3"}
        for v in acts.values():
            assert v.shape[0] == 6

    def test_quantized_extraction_on_grid(self):
        net = make_network()
        x = torch.rand(6, 4)
        params = [None, {"levels": 8, "rounding_point": "center_saturating"}, None]
        acts = net.extract_activations(x, quantizer_params=params)
        levels = torch.linspace(-1, 1, 8).numpy()  # tanh limits
        l2 = acts["L2"].flatten()
        assert all(min(abs(v - levels)) < 1e-6 for v in l2)


class TestConfigRoundTrip:
    def test_round_trip(self):
        net = make_network()
        restored = NeuralNetwork.from_config(net.to_config())
        assert restored.to_config() == net.to_config()
        assert restored.n_layers == net.n_layers

    def test_unknown_net_type_raises(self):
        config = make_network().to_config()
        config["net_type"] = "transformer"
        with pytest.raises(KeyError):
            NeuralNetwork.from_config(config)

    def test_unknown_activation_raises(self):
        infos = make_layer_infos()
        infos[1].activation_function = "swish"
        with pytest.raises(ValueError):
            NeuralNetwork(layer_infos=infos, init_str="xavier")


class TestLimits:
    def test_get_limits_list_follows_activations(self):
        net = make_network()
        limits = net.get_limits_list()
        assert limits == [
            ACTIV_FUNCS_BINNING_LIMITS["input"],
            ACTIV_FUNCS_BINNING_LIMITS["tanh"],
            ACTIV_FUNCS_BINNING_LIMITS["sigmoid"],
        ]


class TestKnownBugs:
    @pytest.mark.xfail(
        reason="known bug: get_layer_sizes builds `shapes` but never returns it "
        "(nninfo/model/neural_network.py:241)",
        strict=True,
    )
    def test_get_layer_sizes_returns_sizes(self):
        net = make_network()
        net.parent = SimpleNamespace(task=SimpleNamespace(x_dim=4))
        sizes = net.get_layer_sizes()
        assert sizes == [4, 3, 2]

    @pytest.mark.xfail(
        raises=AttributeError,
        reason="known bug: NoisyNeuralNetwork.to_config calls `super.to_config()` "
        "without parentheses (nninfo/model/neural_network.py:546)",
        strict=True,
    )
    def test_noisy_network_to_config(self):
        net = NoisyNeuralNetwork(
            layer_infos=make_layer_infos(), init_str="xavier", noise_stddev=0.1
        )
        config = net.to_config()
        assert config["noise_stddev"] == 0.1
