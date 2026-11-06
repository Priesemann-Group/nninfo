"""Tests for nninfo.analysis.lesioning and nninfo.analysis.importance."""

import numpy as np
import pytest
import torch

from nninfo.analysis.lesioning import (
    MeanLesion,
    ShuffleLesion,
    ZeroLesion,
    make_lesion,
)
from nninfo.analysis.importance import activation_importances


IDENTITY = lambda x: x  # noqa: E731


class TestZeroLesion:
    def test_clamps_only_the_target_neuron(self):
        x = torch.rand(64, 5) + 0.1
        out = ZeroLesion(IDENTITY, 2)(x)
        assert torch.all(out[:, 2] == 0)
        assert torch.equal(out[:, [0, 1, 3, 4]], x[:, [0, 1, 3, 4]])

    def test_does_not_mutate_input(self):
        x = torch.rand(16, 5) + 0.1
        before = x.clone()
        ZeroLesion(IDENTITY, 0)(x)
        assert torch.equal(x, before)

    def test_rescale_scales_survivors(self):
        x = torch.rand(8, 5) + 0.1
        out = ZeroLesion(IDENTITY, 0, rescale=True)(x)
        assert torch.allclose(out[:, 1:], x[:, 1:] * (5 / 4))
        assert torch.all(out[:, 0] == 0)

    def test_applies_base_quantizer_first(self):
        from nninfo.model.quantization import Q3Quantizer

        base = Q3Quantizer(n_levels=8, limits=(-1.0, 1.0))
        x = torch.rand(50, 5) * 2 - 1
        out = ZeroLesion(base, 3)(x)
        grid = torch.linspace(-1, 1, 8)
        surv = out[:, [0, 1, 2, 4]].flatten()
        assert torch.all((surv[:, None] - grid[None, :]).abs().min(1).values < 1e-6)


class TestMeanLesion:
    def test_clamps_to_given_value(self):
        x = torch.rand(32, 5)
        out = MeanLesion(IDENTITY, 1, 0.25)(x)
        assert torch.allclose(out[:, 1], torch.full((32,), 0.25))


class TestShuffleLesion:
    def test_preserves_marginal_exactly(self):
        x = torch.rand(500, 5)
        out = ShuffleLesion(IDENTITY, 0, seed=1)(x)
        assert torch.allclose(out[:, 0].sort().values, x[:, 0].sort().values)

    def test_other_neurons_untouched(self):
        x = torch.rand(200, 5)
        out = ShuffleLesion(IDENTITY, 4, seed=2)(x)
        assert torch.equal(out[:, :4], x[:, :4])

    def test_deterministic_given_seed(self):
        x = torch.rand(100, 5)
        a = ShuffleLesion(IDENTITY, 1, seed=7)(x)
        b = ShuffleLesion(IDENTITY, 1, seed=7)(x)
        assert torch.equal(a, b)

    def test_actually_permutes(self):
        x = torch.rand(500, 5)
        out = ShuffleLesion(IDENTITY, 2, seed=3)(x)
        assert not torch.equal(out[:, 2], x[:, 2])


class TestMakeLesion:
    def test_dispatch(self):
        assert isinstance(make_lesion("zero", IDENTITY, 0), ZeroLesion)
        assert isinstance(make_lesion("mean", IDENTITY, 0, mean_value=0.1),
                          MeanLesion)
        assert isinstance(make_lesion("shuffle", IDENTITY, 0), ShuffleLesion)

    def test_mean_requires_value(self):
        with pytest.raises(ValueError):
            make_lesion("mean", IDENTITY, 0)

    def test_unknown_mode(self):
        with pytest.raises(NotImplementedError):
            make_lesion("nearest", IDENTITY, 0)


class TestActivationImportances:
    def test_statistics(self):
        a = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
        out = activation_importances(a, active_threshold=0.5)
        assert out["mean_abs"] == pytest.approx([1.0, 0.25])
        assert out["act_pct"] == pytest.approx([1.0, 0.25])
        assert out["act_std"][0] == pytest.approx(1.0)


class TestCodeObservables:
    """code_observables on canonical systems (cf. Fig. 1d of the paper)."""

    def test_copy_system(self):
        from nninfo.analysis.empirical import code_observables

        y = np.array([0, 1] * 500)
        a = np.stack([y, y, y], axis=1)          # every source == target
        out = code_observables(a, y)
        assert out["total_MI"] == pytest.approx(1.0)
        assert out["deg_red"] == pytest.approx(3.0)
        assert out["deg_vuln"] == pytest.approx(0.0, abs=1e-12)

    def test_xor_system(self):
        from nninfo.analysis.empirical import code_observables

        rng = np.random.default_rng(0)
        x1 = rng.integers(0, 2, 20000)
        x2 = rng.integers(0, 2, 20000)
        y = x1 ^ x2
        out = code_observables(np.stack([x1, x2], axis=1), y)
        assert out["total_MI"] == pytest.approx(1.0, abs=0.01)
        assert out["deg_red"] == pytest.approx(0.0, abs=0.01)
        assert out["deg_vuln"] == pytest.approx(2.0, abs=0.02)

    def test_unique_system(self):
        from nninfo.analysis.empirical import code_observables

        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 20000)
        noise = rng.integers(0, 2, 20000)
        out = code_observables(np.stack([y, noise], axis=1), y)
        assert out["deg_red"] == pytest.approx(1.0, abs=0.01)
        assert out["deg_vuln"] == pytest.approx(1.0, abs=0.02)

    def test_single_source(self):
        from nninfo.analysis.empirical import code_observables

        y = np.array([0, 1] * 500)
        out = code_observables(y[:, None], y)
        assert out["deg_red"] == pytest.approx(1.0)
        assert out["deg_vuln"] == pytest.approx(1.0)

    def test_neurons_subset_selects_surviving_code(self):
        from nninfo.analysis.empirical import code_observables

        y = np.array([0, 1] * 500)
        a = np.stack([y, y, y], axis=1)
        out = code_observables(a, y, neurons=[0, 1])
        assert out["n_sources"] == 2
        assert out["deg_red"] == pytest.approx(2.0)
