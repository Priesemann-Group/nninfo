"""Tests for the information-theoretic math helpers.

Covers PIDMeasurement's static/instance MI and entropy helpers
(nninfo/analysis/pid_measurement.py) and the Reing et al. cohesion
differences (nninfo/analysis/reing.py).
"""

import numpy as np
import pytest
from sxpid import SxPID

from nninfo.analysis.pid_measurement import PIDMeasurement
from nninfo.analysis.reing import compute_C_k, compute_C_k_diffs

# pmf over (S1, S2, T) with T = XOR(S1, S2); uniform independent sources
XOR_PDF = {
    (0, 0, 0): 0.25,
    (0, 1, 1): 0.25,
    (1, 0, 1): 0.25,
    (1, 1, 0): 0.25,
}

# T is a perfect copy of S (one source)
COPY_PDF = {
    (0, 0): 0.5,
    (1, 1): 0.5,
}


class TestEntropyHelper:
    def test_uniform_entropy(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert PIDMeasurement._compute_H(p, p) == pytest.approx(2.0)

    def test_deterministic_entropy(self):
        p = np.array([1.0])
        assert PIDMeasurement._compute_H(p, p) == pytest.approx(0.0)


class TestMarginalizeSameShape:
    def test_marginalize_xor_source(self):
        pdf = SxPID.PDF.from_dict(XOR_PDF)
        # marginalizing out coordinate 0 leaves (S2, T) uniform over 4 combos
        p_s2t = PIDMeasurement.marginalize_same_shape(pdf, 0)
        assert p_s2t == pytest.approx([0.25] * 4)

    def test_marginalize_to_target(self):
        pdf = SxPID.PDF.from_dict(XOR_PDF)
        # marginalizing out both sources leaves p(T): uniform binary -> 0.5 each
        p_t = PIDMeasurement.marginalize_same_shape(pdf, 0, 1)
        assert p_t == pytest.approx([0.5] * 4)


class TestMutualInformation:
    def test_copy_has_one_bit(self):
        pdf = SxPID.PDF.from_dict(COPY_PDF)
        p_st = pdf.probs
        p_s = PIDMeasurement.marginalize_same_shape(pdf, 1)
        p_t = PIDMeasurement.marginalize_same_shape(pdf, 0)
        mi = PIDMeasurement._compute_MI(None, p_st, p_st, p_s, p_t)
        assert mi == pytest.approx(1.0)

    def test_single_xor_source_is_independent_of_target(self):
        pdf = SxPID.PDF.from_dict(XOR_PDF)
        p_full = pdf.probs
        # joint of (S1, T) marginalized to same shape
        p_s1t = PIDMeasurement.marginalize_same_shape(pdf, 1)
        p_s1 = PIDMeasurement.marginalize_same_shape(pdf, 1, 2)
        p_t = PIDMeasurement.marginalize_same_shape(pdf, 0, 1)
        mi = PIDMeasurement._compute_MI(None, p_full, p_s1t, p_s1, p_t)
        assert mi == pytest.approx(0.0, abs=1e-12)

    def test_both_xor_sources_determine_target(self):
        pdf = SxPID.PDF.from_dict(XOR_PDF)
        p_st = pdf.probs
        p_s = PIDMeasurement.marginalize_same_shape(pdf, 2)
        p_t = PIDMeasurement.marginalize_same_shape(pdf, 0, 1)
        mi = PIDMeasurement._compute_MI(None, p_st, p_st, p_s, p_t)
        assert mi == pytest.approx(1.0)


class TestReingCohesion:
    """For XOR: C_0 = H(T) = 1, C_1 = mean_i H(T|S_i) = 1, C_2 = H(T|S1,S2) = 0."""

    def test_compute_C_k_xor(self):
        assert compute_C_k(XOR_PDF, 0) == pytest.approx(1.0)
        assert compute_C_k(XOR_PDF, 1) == pytest.approx(1.0)
        assert compute_C_k(XOR_PDF, 2) == pytest.approx(0.0, abs=1e-12)

    def test_compute_C_k_diffs_xor(self):
        diffs = compute_C_k_diffs(XOR_PDF)
        assert set(diffs.keys()) == {"C(0||1)", "C(1||2)"}
        assert diffs["C(0||1)"] == pytest.approx(0.0, abs=1e-12)
        assert diffs["C(1||2)"] == pytest.approx(1.0)

    def test_compute_C_k_diffs_redundant(self):
        # T copied into both sources: all information is first-order
        pdf = {(0, 0, 0): 0.5, (1, 1, 1): 0.5}
        diffs = compute_C_k_diffs(pdf)
        assert diffs["C(0||1)"] == pytest.approx(1.0)
        assert diffs["C(1||2)"] == pytest.approx(0.0, abs=1e-12)
