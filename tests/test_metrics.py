"""Correctness of src/metrics.py on hand-checkable cases."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import metrics


def test_repeated_split_rmse_is_sqrt_of_mean_mse():
    per_split = pd.DataFrame(
        {"model": "m", "n": 10, "mse": [1.0, 9.0], "mae": [1.0, 3.0], "r2": [0.9, 0.8]}
    )
    out = metrics.aggregate(per_split, ["model"]).iloc[0]
    assert out.rmse == pytest.approx(np.sqrt(5.0))  # not (1 + 3) / 2 = 2
    assert out.mae == pytest.approx(2.0)


def test_point_metrics():
    m = metrics.point([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert m["mse"] == pytest.approx(4 / 3)
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["r2"] == pytest.approx(1 - 4 / 2)


def test_tc_band_metrics_use_10_and_77_kelvin_edges():
    bands = metrics.band_masks([9.99, 10.0, 77.0, 77.01])
    assert list(bands["Tc < 10 K"]) == [True, False, False, False]
    assert list(bands["10-77 K"]) == [False, True, True, False]
    assert list(bands["Tc > 77 K"]) == [False, False, False, True]


def test_subgroups_include_missing_oxygen_rows():
    groups = metrics.subgroup_masks([5, 50, 90], ["other", "cuprate", "cuprate"], [0, 1, 0])
    assert list(groups["oxygen amount missing"]) == [False, True, False]
    assert list(groups["family: cuprate"]) == [False, True, True]
    assert groups["all"].all()


def test_interval_coverage_on_known_case():
    out = metrics.interval([0, 0, 0, 0], [1, 1, 1, 1], [0.5, 1.0, 1.5, -0.1])
    assert out == {"coverage": 0.5, "width": 1.0}


def test_paired_comparison_counts_wins():
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([2.0, 1.0, 4.0])
    out = metrics.paired(a, b)
    assert out["a_better_splits"] == 2 and out["b_better_splits"] == 1
    assert out["mean_diff"] == pytest.approx(-1 / 3)


def test_crps_of_point_mass_equals_absolute_error():
    levels = metrics.QUANTILE_LEVELS
    q = np.full((3, len(levels)), 10.0)
    np.testing.assert_allclose(metrics.crps(q, levels, [7.0, 10.0, 12.5]), [3.0, 0.0, 2.5])


def test_crps_from_grid_matches_closed_form_for_gaussian():
    levels = metrics.QUANTILE_LEVELS
    mu, sigma = 50.0, 8.0
    q = np.tile(stats.norm.ppf(levels, mu, sigma), (4, 1))
    y = np.array([50.0, 42.0, 70.0, 20.0])
    z = (y - mu) / sigma
    exact = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    np.testing.assert_allclose(metrics.crps(q, levels, y), exact, rtol=0.02)


def test_pit_and_exceedance_from_grid():
    levels = metrics.QUANTILE_LEVELS
    q = np.tile(stats.norm.ppf(levels, 50.0, 8.0), (2, 1))
    np.testing.assert_allclose(metrics.cdf_at(q, levels, [50.0, 58.0]), [0.5, 0.8413], atol=0.002)
    # Far beyond the grid the CDF is clamped to the outermost levels.
    np.testing.assert_allclose(metrics.cdf_at(q, levels, [500.0, -500.0]), [0.995, 0.005])


def test_summary_has_expected_columns_and_dtype():
    levels = metrics.QUANTILE_LEVELS
    q = np.tile(np.linspace(0, 100, len(levels)), (5, 1))
    out = metrics.summarize_distribution(np.full(5, 50.0), q, levels, np.full(5, 40.0))
    assert list(out.columns) == [
        "mean",
        "q0.025",
        "q0.05",
        "q0.1",
        "q0.25",
        "q0.5",
        "q0.75",
        "q0.9",
        "q0.95",
        "q0.975",
        "crps",
        "pit",
        "p_above_77K",
    ]
    assert (out.dtypes == "float32").all()
    assert len(metrics.QUANTILE_LEVELS) == 107
