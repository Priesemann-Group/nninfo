"""Tests for quantizers (nninfo/model/quantization.py)."""

import numpy as np
import pytest
import torch

from nninfo.model.quantization import (
    Q3Quantizer,
    StochasticQuantizer,
    quantizer_factory,
    quantizer_list_factory,
)


class TestQ3Quantizer:
    def test_rounds_to_nearest_level(self):
        q = Q3Quantizer(n_levels=2, limits=(0.0, 1.0))
        x = torch.tensor([0.0, 0.2, 0.4999, 0.5001, 0.8, 1.0])
        out = q(x)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        assert torch.allclose(out, expected)

    def test_levels_are_grid_points(self):
        q = Q3Quantizer(n_levels=8, limits=(-1.0, 1.0))
        x = torch.linspace(-1, 1, 101)
        out = q(x)
        grid = torch.linspace(-1, 1, 8)
        # every output value must be (numerically) one of the 8 levels
        dists = (out[:, None] - grid[None, :]).abs().min(dim=1).values
        assert torch.all(dists < 1e-6)

    def test_endpoints_map_to_limits(self):
        q = Q3Quantizer(n_levels=8, limits=(-1.0, 1.0))
        out = q(torch.tensor([-1.0, 1.0]))
        assert torch.allclose(out, torch.tensor([-1.0, 1.0]))

    def test_out_of_limits_raises(self):
        q = Q3Quantizer(n_levels=8, limits=(0.0, 1.0))
        with pytest.raises(AssertionError):
            q(torch.tensor([1.5]))

    def test_infinite_limits_rejected(self):
        with pytest.raises(AssertionError):
            Q3Quantizer(n_levels=8, limits=(0.0, np.inf))


class TestStochasticQuantizer:
    def test_outputs_on_grid(self):
        torch.manual_seed(0)
        q = StochasticQuantizer(n_levels=8, limits=(0.0, 1.0))
        x = torch.rand(1000)
        out = q(x)
        grid = torch.linspace(0, 1, 8)
        dists = (out[:, None] - grid[None, :]).abs().min(dim=1).values
        assert torch.all(dists < 1e-6)

    def test_grid_points_are_fixed_points(self):
        torch.manual_seed(0)
        q = StochasticQuantizer(n_levels=5, limits=(0.0, 1.0))
        grid = torch.linspace(0, 1, 5)
        assert torch.allclose(q(grid), grid)

    def test_unbiased_in_expectation(self):
        torch.manual_seed(0)
        q = StochasticQuantizer(n_levels=2, limits=(0.0, 1.0))
        x = torch.full((20000,), 0.3)
        out = q(x)
        # P(round up to 1) = 0.3 -> mean approx 0.3
        assert abs(out.mean().item() - 0.3) < 0.02

    def test_rounds_to_neighbors_only(self):
        torch.manual_seed(0)
        q = StochasticQuantizer(n_levels=11, limits=(0.0, 1.0))
        x = torch.full((1000,), 0.55)  # between levels 0.5 and 0.6
        out = q(x)
        values = np.unique(out.numpy())
        assert all(min(abs(v - 0.5), abs(v - 0.6)) < 1e-6 for v in values)
        assert len(values) == 2  # both neighbors are hit


class TestQuantizerFactories:
    def test_factory_dispatch(self):
        assert isinstance(
            quantizer_factory("stochastic", 8, (0.0, 1.0)), StochasticQuantizer
        )
        assert isinstance(
            quantizer_factory("center_saturating", 8, (0.0, 1.0)), Q3Quantizer
        )

    def test_factory_unknown_raises(self):
        with pytest.raises(NotImplementedError):
            quantizer_factory("nearest", 8, (0.0, 1.0))

    def test_list_factory_none_gives_identities(self):
        quantizers = quantizer_list_factory(None, [(0, 1), (0, 1), (0, 1)])
        assert len(quantizers) == 3
        x = torch.tensor([0.123])
        for q in quantizers:
            assert q(x) is x

    def test_list_factory_broadcasts_dict(self):
        params = {"levels": 8, "rounding_point": "center_saturating"}
        quantizers = quantizer_list_factory(params, [(0.0, 1.0), (-1.0, 1.0)])
        assert len(quantizers) == 2
        assert all(isinstance(q, Q3Quantizer) for q in quantizers)
        assert quantizers[1].limits == (-1.0, 1.0)

    def test_list_factory_per_layer_none_entries(self):
        params = [None, {"levels": 8, "rounding_point": "center_saturating"}, None]
        limits = [None, (0.0, 1.0), None]  # None-quantizer entries never touch limits
        quantizers = quantizer_list_factory(params, limits)
        x = torch.tensor([0.123])
        assert quantizers[0](x) is x
        assert isinstance(quantizers[1], Q3Quantizer)
        assert quantizers[2](x) is x
