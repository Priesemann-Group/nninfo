"""Tests for DataSet / SubSet / CachedDataset / LazyDataset (nninfo/data_set.py)."""

import numpy as np
import pytest
import torch

from nninfo.data_set import CachedDataset, LazyDataset, SubSet
from nninfo.tasks.task import Task


@pytest.fixture(scope="module")
def tishby_full_set():
    return CachedDataset(Task.from_id("tishby_dat"), "full_set")


class TestCachedDataset:
    def test_len_and_getitem(self, tishby_full_set):
        assert len(tishby_full_set) == 4096
        x, y = tishby_full_set[0]
        assert x.shape == (12,)
        assert y.shape == (1,)


class TestRandomSplit:
    def test_split_lengths(self):
        ds = CachedDataset(Task.from_id("tishby_dat"), "full_set")
        names = ds.train_test_val_random_split(2796, 1300, 0, seed=42)
        assert names == ["full_set/train", "full_set/test", "full_set/val"]
        assert len(ds.find("full_set/train")) == 2796
        assert len(ds.find("full_set/test")) == 1300
        assert len(ds.find("full_set/val")) == 0

    def test_split_is_disjoint_and_complete(self):
        ds = CachedDataset(Task.from_id("tishby_dat"), "full_set")
        ds.train_test_val_random_split(2796, 1300, 0, seed=42)
        train = ds.find("full_set/train").indices
        test = ds.find("full_set/test").indices
        combined = np.concatenate([train, test])
        assert len(np.unique(combined)) == 4096  # disjoint and covering

    def test_split_deterministic_in_seed(self):
        ds1 = CachedDataset(Task.from_id("tishby_dat"), "full_set")
        ds2 = CachedDataset(Task.from_id("tishby_dat"), "full_set")
        ds1.train_test_val_random_split(2796, 1300, 0, seed=42)
        ds2.train_test_val_random_split(2796, 1300, 0, seed=42)
        assert np.array_equal(
            ds1.find("full_set/train").indices, ds2.find("full_set/train").indices
        )

    def test_wrong_total_raises(self):
        ds = CachedDataset(Task.from_id("tishby_dat"), "full_set")
        with pytest.raises(ValueError):
            ds.train_test_val_random_split(10, 10, 0, seed=1)

    def test_subset_getitem_maps_to_parent(self):
        ds = CachedDataset(Task.from_id("tishby_dat"), "full_set")
        ds.train_test_val_random_split(2796, 1300, 0, seed=42)
        subset = ds.find("full_set/train")
        x_sub, y_sub = subset[0]
        x_parent, y_parent = ds[subset.indices[0]]
        assert torch.equal(x_sub, x_parent)
        assert torch.equal(y_sub, y_parent)


class TestFind:
    def test_find_self_and_missing(self, tishby_full_set):
        assert tishby_full_set.find("full_set") is tishby_full_set
        assert tishby_full_set.find("nonexistent") is None


class TestSubsetConfigRoundTrip:
    def test_slice_subset_round_trip(self):
        ds = CachedDataset(Task.from_id("tishby_dat"), "full_set")
        ds.train_test_val_random_split(2796, 1300, 0, seed=42)
        original = ds.find("full_set/train")

        restored = SubSet.from_config(ds, original.to_config())
        assert restored.name == original.name
        assert np.array_equal(restored.indices, original.indices)

    @pytest.mark.xfail(
        reason="known bug: nested subsets serialize via nonexistent to_dict() "
        "(nninfo/data_set.py:212)",
        strict=True,
    )
    def test_nested_subset_round_trip(self):
        ds = CachedDataset(Task.from_id("tishby_dat"), "full_set")
        ds.train_test_val_random_split(2796, 1300, 0, seed=42)
        train = ds.find("full_set/train")
        train.create_subset("full_set/train/sub", slice(None, 100))
        config = train.to_config()
        assert config["subsets"][0]["name"] == "full_set/train/sub"


class TestLazyDataset:
    def test_len_and_generation(self, xor_task):
        ds = LazyDataset(xor_task, "full_set", length=10, seed=1234)
        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape == (2,)
        with pytest.raises(IndexError):
            ds[10]

    def test_same_seed_same_samples(self, xor_task):
        ds1 = LazyDataset(xor_task, "a", length=5, seed=1234)
        ds2 = LazyDataset(xor_task, "b", length=5, seed=1234)
        for i in range(5):
            x1, _ = ds1[i]
            x2, _ = ds2[i]
            assert np.array_equal(x1, x2)

    def test_indexing_is_stateless(self, xor_task):
        # accessing the same index twice must give the same sample
        ds = LazyDataset(xor_task, "a", length=5, seed=1234)
        x_first, _ = ds[3]
        ds[1]
        x_again, _ = ds[3]
        assert np.array_equal(x_first, x_again)

    @pytest.mark.xfail(
        reason="known bug: with seed=None the generated seed is stored but not "
        "used (Philox(seed) instead of Philox(self._seed), "
        "nninfo/data_set.py:264), so the saved config does not reproduce data",
        strict=True,
    )
    def test_generated_seed_is_actually_used(self, xor_task):
        ds = LazyDataset(xor_task, "a", length=5, seed=None)
        clone = LazyDataset(xor_task, "b", length=5, seed=ds._seed)
        x1, _ = ds[0]
        x2, _ = clone[0]
        assert np.array_equal(x1, x2)
