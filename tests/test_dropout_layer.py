"""One-neuron dropout as a network layer.

One-neuron dropout is a connection layer of its own (`one_neuron_dropout`), so
that the activations before and after the intervention are separately labelled
and measurable.

The dropout networks of the Shannon observables paper were trained with an
earlier implementation that zeroed the unit inside the layer's *quantizer*.
`ReferenceQuantizerDropout` below reproduces that implementation as a test
oracle, so that the layer provably retrains those networks exactly: same seed,
same weights, same trajectory, down to the last bit.
"""

import copy

import pytest
import torch
import torch.nn as nn

import nninfo
from nninfo.model.neural_network import (
    ACTIV_FUNCS_BINNING_LIMITS,
    NeuralNetwork,
    OneNeuronDropout,
)
from nninfo.model.quantization import (
    StochasticQuantizer,
    quantizer_list_factory,
)

LEVELS = 8
TANH_LIMITS = (-1.0, 1.0)


class ReferenceQuantizerDropout:
    """The quantizer-wrapper implementation the paper's networks were trained with."""

    def __init__(self, base_quantizer, p: float = 1.0):
        self.base_quantizer = base_quantizer
        self.p = float(p)

    def __call__(self, x):
        x = self.base_quantizer(x)
        rows = torch.arange(x.shape[0], device=x.device)
        dropped = torch.randint(0, x.shape[1], (x.shape[0],), device=x.device)
        mask = torch.ones_like(x)
        mask[rows, dropped] = 0.0
        if self.p < 1.0:
            apply = torch.rand(x.shape[0], device=x.device) < self.p
            mask = torch.where(apply.unsqueeze(1), mask, torch.ones_like(x))
        return x * mask


def linear(n_in, n_out, activation="tanh"):
    return nninfo.LayerInfo(
        connection_layer="linear",
        connection_layer_kwargs={"in_features": n_in, "out_features": n_out},
        activation_function=activation,
    )


def dropout_layer(p=1.0):
    return nninfo.LayerInfo(
        connection_layer="one_neuron_dropout",
        connection_layer_kwargs={"p": p},
        activation_function=None,
    )


def plain_layer_infos():
    """784-50-10-5-5-5-10, the original MNIST net (no dropout entry)."""
    return [
        nninfo.LayerInfo(connection_layer="input", activation_function="input"),
        linear(784, 50), linear(50, 10), linear(10, 5), linear(5, 5),
        linear(5, 5), linear(5, 10, "softmax_output"),
    ]


def layer_infos_with_dropout_after(index, p=1.0):
    """Same net with a dropout layer inserted right after layer_infos[index]."""
    infos = plain_layer_infos()
    infos.insert(index + 1, dropout_layer(p))
    return infos


def stochastic_quantizers(n_entries, dropout_index=None, p=1.0):
    """Quantizer list for the training path: stochastic on the hidden layers.

    With `dropout_index` set, that entry additionally gets the reference
    wrapper -- this is the oracle configuration, not the production one.
    """
    qs = [(lambda x: x)]
    for _ in range(n_entries - 2):
        qs.append(StochasticQuantizer(LEVELS, TANH_LIMITS))
    qs.append(lambda x: x)
    if dropout_index is not None:
        qs[dropout_index] = ReferenceQuantizerDropout(
            StochasticQuantizer(LEVELS, TANH_LIMITS), p=p)
    return qs


def train_steps(net, quantizers, n_steps=5, batch=8, seed=1234):
    """A few SGD steps on a fixed synthetic batch, in train mode."""
    torch.manual_seed(seed)
    x = torch.rand(batch, 784)
    y = torch.randint(0, 10, (batch,))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    net.train()
    losses = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        out = net.forward(x, quantizers=quantizers)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


class TestBitPerfectEquivalence:
    """The layer must reproduce the quantizer wrapper exactly, not just closely."""

    def _build_pair(self, p=1.0, drop_after=3, init_seed=7):
        torch.manual_seed(init_seed)
        old = NeuralNetwork(layer_infos=plain_layer_infos(), init_str="xavier")
        torch.manual_seed(init_seed)
        new = NeuralNetwork(layer_infos=layer_infos_with_dropout_after(drop_after, p),
                            init_str="xavier")
        old_qs = stochastic_quantizers(7, dropout_index=drop_after, p=p)
        new_qs = quantizer_list_factory(
            [None] + 3 * [{"levels": LEVELS, "rounding_point": "stochastic"}]
            + [None]  # the dropout layer is not quantized
            + 2 * [{"levels": LEVELS, "rounding_point": "stochastic"}] + [None],
            new.get_limits_list())
        return old, old_qs, new, new_qs

    def test_the_comparison_is_not_vacuous(self):
        """Guard: dropout must actually change the trajectory.

        Without this, the equivalence tests above would also pass if the
        intervention were silently inactive in both implementations.
        """
        _, _, new, new_qs = self._build_pair()
        torch.manual_seed(7)
        plain = NeuralNetwork(layer_infos=plain_layer_infos(), init_str="xavier")
        plain_qs = stochastic_quantizers(7)  # no dropout anywhere

        torch.manual_seed(4242)
        with_dropout = train_steps(new, new_qs)
        torch.manual_seed(4242)
        without = train_steps(plain, plain_qs)
        assert with_dropout != without

    def test_weight_init_is_identical(self):
        """The extra layer must not perturb the weight-init RNG stream."""
        old, _, new, _ = self._build_pair()
        for a, b in zip(old.parameters(), new.parameters()):
            assert torch.equal(a, b)

    def test_forward_is_bit_identical(self):
        old, old_qs, new, new_qs = self._build_pair()
        old.train()
        new.train()
        x = torch.rand(16, 784)

        torch.manual_seed(99)
        out_old = old.forward(x, quantizers=old_qs)
        torch.manual_seed(99)
        out_new = new.forward(x, quantizers=new_qs)
        assert torch.equal(out_old, out_new)

    def test_training_trajectory_is_bit_identical(self):
        old, old_qs, new, new_qs = self._build_pair()
        torch.manual_seed(4242)
        losses_old = train_steps(old, old_qs)
        torch.manual_seed(4242)
        losses_new = train_steps(new, new_qs)
        assert losses_old == losses_new
        for a, b in zip(old.parameters(), new.parameters()):
            assert torch.equal(a, b)

    @pytest.mark.parametrize("p", [0.5, 0.25])
    def test_partial_dropout_trajectory_is_bit_identical(self, p):
        old, old_qs, new, new_qs = self._build_pair(p=p)
        torch.manual_seed(4242)
        losses_old = train_steps(old, old_qs)
        torch.manual_seed(4242)
        losses_new = train_steps(new, new_qs)
        assert losses_old == losses_new

    @pytest.mark.parametrize("drop_after", [3, 4, 5])
    def test_every_bottleneck_position_matches(self, drop_after):
        old, old_qs, new, _ = self._build_pair(drop_after=drop_after)
        # the inserted entry sits at drop_after + 1 and is not quantized
        params = [None] + 5 * [{"levels": LEVELS, "rounding_point": "stochastic"}] + [None]
        params.insert(drop_after + 1, None)
        new_qs = quantizer_list_factory(params, new.get_limits_list())

        torch.manual_seed(4242)
        losses_old = train_steps(old, old_qs)
        torch.manual_seed(4242)
        losses_new = train_steps(new, new_qs)
        assert losses_old == losses_new


class TestCheckpointCompatibility:
    """Adding the layer must not rename any state_dict key."""

    def test_state_dict_keys_unchanged(self):
        plain = NeuralNetwork(layer_infos=plain_layer_infos(), init_str="xavier")
        with_dropout = NeuralNetwork(
            layer_infos=layer_infos_with_dropout_after(3), init_str="xavier")
        assert list(plain.state_dict()) == list(with_dropout.state_dict())

    def test_dropout_module_contributes_no_state(self):
        net = NeuralNetwork(layer_infos=layer_infos_with_dropout_after(3),
                            init_str="xavier")
        assert not list(net._aux_module_list.state_dict())
        assert len(net._aux_module_list) == 1
        assert len(net._module_list) == 6  # the six Linear layers, unshifted

    def test_old_checkpoint_loads_strictly(self):
        plain = NeuralNetwork(layer_infos=plain_layer_infos(), init_str="xavier")
        with_dropout = NeuralNetwork(
            layer_infos=layer_infos_with_dropout_after(3), init_str="xavier")
        with_dropout.load_state_dict(copy.deepcopy(plain.state_dict()), strict=True)
        for a, b in zip(plain.parameters(), with_dropout.parameters()):
            assert torch.equal(a, b)


class TestTrainEvalGating:
    """Dropout is a training-time intervention, inert under eval()."""

    def test_drops_exactly_one_neuron_in_train_mode(self):
        torch.manual_seed(0)
        layer = OneNeuronDropout(p=1.0)
        layer.train()
        out = layer(torch.rand(256, 5) + 0.1)
        assert torch.all((out == 0).sum(dim=1) == 1)

    def test_identity_in_eval_mode(self):
        torch.manual_seed(0)
        layer = OneNeuronDropout(p=1.0)
        layer.eval()
        x = torch.rand(256, 5) + 0.1
        assert torch.equal(layer(x), x)

    def test_eval_consumes_no_randomness(self):
        """Measurement passes must not perturb the RNG stream."""
        layer = OneNeuronDropout(p=1.0)
        layer.eval()
        x = torch.rand(64, 5) + 0.1
        torch.manual_seed(3)
        expected = torch.rand(4)
        torch.manual_seed(3)
        layer(x)
        assert torch.equal(torch.rand(4), expected)

    def test_network_train_eval_propagates_to_the_layer(self):
        net = NeuralNetwork(layer_infos=layer_infos_with_dropout_after(3),
                            init_str="xavier")
        layer = net._connection_layers[4]
        assert isinstance(layer, OneNeuronDropout)
        net.eval()
        assert not layer.training
        net.train()
        assert layer.training

    def test_measurement_path_sees_the_intact_code(self):
        """In eval mode the pre- and post-dropout labels agree."""
        net = NeuralNetwork(layer_infos=layer_infos_with_dropout_after(3),
                            init_str="xavier")
        qs = quantizer_list_factory(None, net.get_limits_list())
        net.eval()
        _, activations = net.forward(torch.rand(32, 784), quantizers=qs,
                                     return_activations=True)
        assert (activations["L4"] == activations["L5"]).all()

    def test_survivors_are_not_rescaled(self):
        torch.manual_seed(0)
        layer = OneNeuronDropout(p=1.0)
        layer.train()
        x = torch.rand(64, 5) + 0.1
        out = layer(x)
        survivors = out != 0
        assert torch.equal(out[survivors], x[survivors])

    def test_partial_dropout_fraction(self):
        torch.manual_seed(0)
        layer = OneNeuronDropout(p=0.5)
        layer.train()
        out = layer(torch.rand(20_000, 5) + 0.1)
        zeros = (out == 0).sum(dim=1)
        assert set(zeros.unique().tolist()) == {0, 1}
        assert 0.48 < (zeros == 1).float().mean().item() < 0.52

    def test_gradient_flows_through_survivors_only(self):
        torch.manual_seed(0)
        layer = OneNeuronDropout(p=1.0)
        layer.train()
        x = (torch.rand(32, 5) * 0.9 + 0.1).detach().requires_grad_()
        layer(x).sum().backward()
        assert torch.all((x.grad == 0).sum(dim=1) == 1)
        assert torch.all((x.grad == 1).sum(dim=1) == 4)


class TestConfigRoundTrip:
    """The intervention is part of the network config, so it round-trips."""

    def test_layer_info_round_trip(self):
        info = dropout_layer(0.5)
        restored = nninfo.LayerInfo.from_config(info.to_config())
        assert restored == info
        assert restored.connection_layer_kwargs["p"] == 0.5

    def test_network_config_round_trip_rebuilds_the_layer(self):
        net = NeuralNetwork(layer_infos=layer_infos_with_dropout_after(3, p=0.5),
                            init_str="xavier")
        restored = NeuralNetwork.from_config(net.to_config())
        layer = restored._connection_layers[4]
        assert isinstance(layer, OneNeuronDropout)
        assert layer.p == 0.5

    def test_limits_are_inherited_from_the_preceding_layer(self):
        net = NeuralNetwork(layer_infos=layer_infos_with_dropout_after(3),
                            init_str="xavier")
        limits = net.get_limits_list()
        assert limits[4] == limits[3] == ACTIV_FUNCS_BINNING_LIMITS["tanh"]

    def test_plain_network_limits_unchanged(self):
        net = NeuralNetwork(layer_infos=plain_layer_infos(), init_str="xavier")
        assert net.get_limits_list() == [
            ACTIV_FUNCS_BINNING_LIMITS["input"],
            *5 * [ACTIV_FUNCS_BINNING_LIMITS["tanh"]],
            ACTIV_FUNCS_BINNING_LIMITS["softmax_output"],
        ]
