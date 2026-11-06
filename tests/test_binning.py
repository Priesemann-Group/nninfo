"""Tests for Binning (nninfo/analysis/binning.py)."""

import numpy as np
import pytest

from nninfo.analysis.binning import Binning
from nninfo.model.neural_network import NeuronID


def make_none_binning(groupings):
    return Binning.from_binning_method("none", groupings, None, None)


class TestFromBinningMethod:
    def test_dispatch_none(self):
        b = make_none_binning([[NeuronID("L1", (1,))]])
        assert b.get_binning_method() == "none"

    def test_unknown_method_raises(self):
        with pytest.raises(NotImplementedError):
            Binning.from_binning_method("magic", [], None, None)


class TestNoneBinning:
    def test_pdf_from_counts(self):
        s1 = NeuronID("L1", (1,))
        t = NeuronID("Y", (1,))
        binning = make_none_binning([[s1], [t]])

        activations = {
            s1: np.array([0, 0, 1, 1]),
            t: np.array([0, 1, 1, 1]),
        }
        binning.apply(activations)
        pdf = binning.get_pdf()

        # realizations are ((source_tuple), (target_tuple)) per sample
        assert pdf[((0,), (0,))] == pytest.approx(0.25)
        assert pdf[((0,), (1,))] == pytest.approx(0.25)
        assert pdf[((1,), (1,))] == pytest.approx(0.5)
        assert sum(pdf.values()) == pytest.approx(1.0)

    def test_multi_neuron_grouping(self):
        s1, s2 = NeuronID("L1", (1,)), NeuronID("L1", (2,))
        t = NeuronID("Y", (1,))
        binning = make_none_binning([[s1, s2], [t]])

        binning.apply(
            {
                s1: np.array([0, 1]),
                s2: np.array([1, 0]),
                t: np.array([1, 1]),
            }
        )
        pdf = binning.get_pdf()
        assert pdf[((0, 1), (1,))] == pytest.approx(0.5)
        assert pdf[((1, 0), (1,))] == pytest.approx(0.5)

    def test_accumulates_over_batches(self):
        s1 = NeuronID("L1", (1,))
        t = NeuronID("Y", (1,))
        binning = make_none_binning([[s1], [t]])

        batch = {s1: np.array([0]), t: np.array([1])}
        binning.apply(batch)
        binning.apply(batch)
        pdf = binning.get_pdf()
        assert pdf[((0,), (1,))] == pytest.approx(1.0)

    def test_reset_clears_counts(self):
        s1 = NeuronID("L1", (1,))
        t = NeuronID("Y", (1,))
        binning = make_none_binning([[s1], [t]])
        binning.apply({s1: np.array([0]), t: np.array([1])})
        binning.reset()
        binning.apply({s1: np.array([1]), t: np.array([0])})
        pdf = binning.get_pdf()
        assert pdf == {((1,), (0,)): pytest.approx(1.0)}


class TestFixedSizeBinning:
    def make(self, limits_dict, n_bins, groupings):
        return Binning.from_binning_method(
            "fixed_size", groupings, None, None, limits=limits_dict, n_bins=n_bins
        )

    def test_binary_limits_pass_through_as_int(self):
        t = NeuronID("Y", (1,))
        binning = self.make({t: "binary"}, 4, [[t]])
        out = binning.discretize(np.array([0.0, 1.0]), t)
        assert out.dtype == np.int64
        assert list(out) == [0, 1]

    def test_unit_range_binning(self):
        s = NeuronID("L1", (1,))
        binning = self.make({s: (0.0, 1.0)}, 4, [[s]])
        out = binning.discretize(np.array([0.0, 0.1, 0.3, 0.6, 0.9]), s)
        assert list(out) == [0, 0, 1, 2, 3]

    def test_infinite_limits_raise(self):
        s = NeuronID("L1", (1,))
        binning = self.make({s: (0.0, np.inf)}, 4, [[s]])
        with pytest.raises(NotImplementedError):
            binning.discretize(np.array([0.5]), s)

    def test_offset_range_even_bins_is_pure_relabeling(self):
        # discretize ignores limits[0], producing shifted/negative bin indices.
        # Bin labels are opaque symbols to the downstream Counter/pmf, so as
        # long as limits[0] is an integer multiple of the bin width (e.g. even
        # n_bins on tanh's (-1,1)) this is a bijective relabeling with zero
        # effect on any information quantity.
        s = NeuronID("L1", (1,))
        binning = self.make({s: (-1.0, 1.0)}, 4, [[s]])
        x = np.linspace(-1.0, 1.0, 10001)
        actual = binning.discretize(x, s)
        intended = np.floor((x - (-1.0)) / (2.0 / 4)).astype(int)
        pairs = set(zip(actual.tolist(), intended.tolist()))
        assert len(pairs) == len({a for a, _ in pairs}) == len({i for _, i in pairs})

    @pytest.mark.xfail(
        reason="known bug: discretize ignores limits[0] "
        "(nninfo/analysis/binning.py:128). When limits[0] is not an integer "
        "multiple of the bin width (e.g. odd n_bins on tanh's (-1,1)), the "
        "partition itself shifts — boundaries fall at (-2/3, 0, 2/3) instead "
        "of (-1/3, 1/3, 1) — which is NOT a relabeling and changes the "
        "empirical pmf. Even-n_bins and (0,1) ranges are unaffected.",
        strict=True,
    )
    def test_offset_range_odd_bins_partition(self):
        s = NeuronID("L1", (1,))
        binning = self.make({s: (-1.0, 1.0)}, 3, [[s]])
        x = np.linspace(-1.0, 1.0, 10001)
        actual = binning.discretize(x, s)
        intended = np.floor((x - (-1.0)) / (2.0 / 3)).astype(int)
        pairs = set(zip(actual.tolist(), intended.tolist()))
        assert len(pairs) == len({a for a, _ in pairs}) == len({i for _, i in pairs})
