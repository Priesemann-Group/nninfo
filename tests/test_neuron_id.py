"""Tests for NeuronID and LayerInfo (nninfo/model/neural_network.py)."""

import yaml

from nninfo.file_io import NoAliasDumper
from nninfo.model.neural_network import LayerInfo, NeuronID


class TestNeuronID:
    def test_equality_and_hashability(self):
        a = NeuronID("L1", (3,))
        b = NeuronID("L1", (3,))
        c = NeuronID("L2", (3,))
        assert a == b
        assert a != c
        assert len({a, b, c}) == 2  # frozen dataclass -> usable as dict key

    def test_yaml_round_trip_simple_index(self):
        original = NeuronID("Y", (1,))
        dumped = yaml.dump(original, Dumper=NoAliasDumper)
        loaded = yaml.safe_load(dumped)
        assert loaded == original

    def test_yaml_round_trip_multidim_index(self):
        original = NeuronID("L2", (3, 4))
        loaded = yaml.safe_load(yaml.dump(original, Dumper=NoAliasDumper))
        assert loaded == original

    def test_yaml_round_trip_in_container(self):
        # PID configs store lists of NeuronIDs
        original = [[NeuronID("L1", (1,)), NeuronID("L1", (2,))], [NeuronID("Y", (1,))]]
        loaded = yaml.safe_load(yaml.dump(original, Dumper=NoAliasDumper))
        assert loaded == original


class TestLayerInfo:
    def test_config_round_trip(self):
        original = LayerInfo(
            connection_layer="linear",
            activation_function="tanh",
            connection_layer_kwargs={"in_features": 12, "out_features": 4},
        )
        restored = LayerInfo.from_config(original.to_config())
        assert restored == original

    def test_default_kwargs_are_independent(self):
        # mutable default fields must not be shared between instances
        a = LayerInfo(connection_layer="input", activation_function="input")
        b = LayerInfo(connection_layer="input", activation_function="input")
        a.connection_layer_kwargs["x"] = 1
        assert b.connection_layer_kwargs == {}
