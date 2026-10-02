import numpy as np

from src import benchmark


def test_learning_curve_subsets_are_nested_seeded_and_train_only():
    is_test = np.zeros(1000, dtype=bool)
    is_test[::3] = True
    small = benchmark.learning_curve_rows(is_test, split=2, n=100)
    large = benchmark.learning_curve_rows(is_test, split=2, n=300)
    assert len(small) == 100 and len(large) == 300
    assert set(small) <= set(large)
    assert not is_test[large].any()
    np.testing.assert_array_equal(small, benchmark.learning_curve_rows(is_test, split=2, n=100))
    assert not np.array_equal(small, benchmark.learning_curve_rows(is_test, split=3, n=100))


def test_machine_info_names_the_host_and_versions():
    info = benchmark.machine_info()
    assert info["host"] and info["cpu_count"] >= 1
    assert {"platform", "python", "xgboost", "numpy"} <= set(info)
