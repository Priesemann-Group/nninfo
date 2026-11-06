"""Tests for Schedule (nninfo/schedule.py)."""

import numpy as np

from nninfo.schedule import Schedule


class TestLogSpacedChapters:
    def test_basic_properties(self):
        s = Schedule()
        s.create_log_spaced_chapters(n_epochs=1000, n_chapters_wished=30)

        ends = s.chapter_ends
        assert ends[0] == 0  # prepended zero
        assert ends[1] == 1  # 10**start_exp
        assert ends[-1] == 1000
        assert ends == sorted(ends)
        assert len(ends) == len(set(ends))  # unique after rounding

    def test_continued_extends_chapter_ends(self):
        s = Schedule()
        s.create_log_spaced_chapters(n_epochs=100, n_chapters_wished=10)
        n = len(s.chapter_ends)
        assert s.chapter_ends_continued[:n] == s.chapter_ends
        assert s.chapter_ends_continued[-1] == 100 * 100  # n_epochs**2

    def test_fractional_keeps_float_epochs(self):
        s = Schedule()
        s.create_log_spaced_chapters(n_epochs=10, n_chapters_wished=5, fractional=True)
        assert any(e != int(e) for e in s.chapter_ends)

    def test_get_epoch_for_chapter(self):
        s = Schedule()
        s.create_log_spaced_chapters(n_epochs=1000, n_chapters_wished=30)
        assert s.get_epoch_for_chapter(0) == 0
        assert s.get_epoch_for_chapter(1) == s.chapter_ends[1]


class TestLinSpacedChapters:
    def test_basic_properties(self):
        s = Schedule()
        s.create_lin_spaced_chapters(n_epochs=100, n_chapters_wished=10)
        assert s.chapter_ends == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    def test_continued(self):
        s = Schedule()
        s.create_lin_spaced_chapters(n_epochs=100, n_chapters_wished=10)
        assert s.chapter_ends_continued[-1] == 100 * 100


class TestConfigRoundTrip:
    def test_round_trip(self):
        s = Schedule()
        s.create_log_spaced_chapters(n_epochs=1000, n_chapters_wished=30)
        restored = Schedule.from_config(s.to_config())
        assert restored.chapter_ends == s.chapter_ends
        assert restored.chapter_ends_continued == s.chapter_ends_continued
