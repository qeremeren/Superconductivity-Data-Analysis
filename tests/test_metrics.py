"""Correctness of src/metrics.py on hand-checkable cases (Phase 2-3)."""

import pytest

pytestmark = pytest.mark.skip(reason="src/metrics.py is implemented in Phases 2-3")


def test_repeated_split_rmse_is_sqrt_of_mean_mse():
    """The paper's definition: sqrt(mean over repeats of MSE), not mean of RMSEs."""


def test_tc_band_metrics_use_10_and_77_kelvin_edges(): ...


def test_interval_coverage_on_known_case(): ...


def test_crps_of_point_mass_equals_absolute_error(): ...


def test_crps_of_bar_distribution_matches_numerical_integral(): ...
