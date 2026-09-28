"""Leakage and reproducibility guarantees for src/splits.py (Phase 2)."""

import pytest

pytestmark = pytest.mark.skip(reason="src/splits.py is implemented in Phase 2")


def test_grouped_split_never_puts_a_formula_in_both_train_and_test(): ...


def test_random_split_follows_paper_protocol():
    """2/3 train, 1/3 test, 25 repeats, as in Hamidieh (2018)."""


def test_leave_family_out_holds_out_whole_families():
    """Cuprates (Cu>0 and O>0), iron-based (Fe>0 and (As>0 or Se>0)), everything else."""


def test_splits_reproduce_from_seed(): ...


def test_saved_splits_match_regenerated_splits():
    """data/splits/ on disk must equal what splits.py produces from the same seeds."""
