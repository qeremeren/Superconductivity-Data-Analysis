"""Leakage and reproducibility guarantees for src/splits.py."""

import numpy as np
import pytest

from src import data, splits

needs_data = pytest.mark.skipif(
    not data.is_downloaded(), reason="data/raw/ missing or unverified; run `make data`"
)


def _no_group_on_both_sides(groups, masks):
    groups = np.asarray(groups)
    for is_test in masks:
        assert not set(groups[is_test]) & set(groups[~is_test])


def test_grouped_split_never_puts_a_group_in_both_train_and_test():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 300, size=2000)
    _no_group_on_both_sides(groups, splits.grouped_splits(groups))


def test_random_split_follows_paper_protocol():
    masks = splits.random_splits(data.N_ROWS)
    assert masks.shape == (25, data.N_ROWS)
    assert (masks.sum(axis=1) == 7088).all()  # 21,263 - round(2/3 * 21,263)
    assert len({m.tobytes() for m in masks}) == 25


def test_grouped_split_matches_random_proportions():
    groups = np.repeat(np.arange(500), np.random.default_rng(1).integers(1, 20, size=500))
    masks = splits.grouped_splits(groups)
    max_group = np.bincount(groups).max()
    target = len(groups) / 3
    assert (np.abs(masks.sum(axis=1) - target) <= max_group).all()


def test_splits_reproduce_from_seed():
    groups = np.random.default_rng(2).integers(0, 100, size=1000)
    np.testing.assert_array_equal(splits.grouped_splits(groups), splits.grouped_splits(groups))
    np.testing.assert_array_equal(splits.random_splits(1000), splits.random_splits(1000))


@needs_data
def test_saved_splits_match_regenerated_splits():
    for kind, masks in splits.build_all().items():
        np.testing.assert_array_equal(splits.load(kind), masks)


@needs_data
def test_grouped_split_has_no_material_leakage_in_data():
    groups = data.composition_key(data.load_unique_m())
    _no_group_on_both_sides(groups, splits.load("grouped"))


@needs_data
def test_oxygen_variants_never_cross_in_grouped_no_oxygen():
    um = data.load_unique_m()
    masks = splits.load("grouped_no_oxygen")
    _no_group_on_both_sides(data.composition_key_without_oxygen(um), masks)
    _no_group_on_both_sides(data.composition_key(um), masks)  # implied, but check it
    assert np.allclose(masks.mean(axis=1), 1 / 3, atol=0.01)


@pytest.mark.skip(reason="leave-family-out split is added in Phase 2")
def test_leave_family_out_holds_out_whole_families():
    """Cuprates (Cu>0 and O>0), iron-based (Fe>0 and (As>0 or Se>0)), everything else."""
