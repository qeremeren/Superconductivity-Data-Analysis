import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import budget, metrics, models


def _job(seed=0, n_train=40, n_test=10, name="grouped/composition/split_00"):
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame(rng.normal(size=(n_train, 3)), columns=["a", "b", "c"])
    X_test = pd.DataFrame(rng.normal(size=(n_test, 3)), columns=["a", "b", "c"])
    return models.TabPFNJob(
        name=name,
        seed=seed,
        X_train=X_train,
        y_train=rng.uniform(0, 100, n_train),
        X_test=X_test,
        y_test=rng.uniform(0, 100, n_test),
        test_rows=np.arange(100, 100 + n_test),
    )


class FakePredict:
    def __init__(self):
        self.calls = 0

    def __call__(self, X_train, y_train, X_test, seed):
        self.calls += 1
        q = np.tile(stats.norm.ppf(metrics.QUANTILE_LEVELS, 50, 10), (len(X_test), 1))
        return {"mean": np.full(len(X_test), 50.0), "quantiles": q, "meta": {"m": 1}, "timings": {}}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "RAW_CACHE", tmp_path / "raw")
    return models.TabPFNCache(tmp_path / "out")


@pytest.fixture
def run_budget(tmp_path):
    return budget.RunBudget("p2_grouped", 5, 10_000, tmp_path / "ledger.jsonl")


def test_missing_cache_never_calls_the_api(cache):
    fake = FakePredict()
    with pytest.raises(models.MissingCache, match="never calls the API"):
        cache.get(_job(), live=False, predict=fake)
    assert fake.calls == 0


def test_live_call_charges_once_then_serves_from_cache(cache, run_budget):
    fake, job = FakePredict(), _job()
    first = cache.get(job, live=True, budget=run_budget, predict=fake)
    second = cache.get(job, live=True, budget=run_budget, predict=fake)
    assert fake.calls == 1 and run_budget.used == 1
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns[:3]) == ["row", "y", "mean"]
    assert cache.raw_path(job).is_file() and cache.record_path(job).is_file()


def test_changed_inputs_are_refused_as_stale(cache, run_budget):
    cache.get(_job(seed=0), live=True, budget=run_budget, predict=FakePredict())
    changed = _job(seed=1)  # same name, different data and seed
    with pytest.raises(models.StaleCache):
        cache.get(changed, live=True, budget=run_budget, predict=FakePredict())


def test_raw_grid_is_resummarized_without_a_request(cache, run_budget):
    fake, job = FakePredict(), _job()
    first = cache.get(job, live=True, budget=run_budget, predict=fake)
    cache.summary_path(job).unlink()
    again = cache.get(job, live=False, predict=fake)
    assert fake.calls == 1
    pd.testing.assert_frame_equal(first, again)


def test_live_call_needs_a_budget(cache):
    with pytest.raises(ValueError, match="RunBudget"):
        cache.get(_job(), live=True, budget=None, predict=FakePredict())


def test_inner_validation_keeps_groups_whole():
    groups = np.repeat(np.arange(200), 5)
    is_val = models.inner_validation_mask(len(groups), groups, seed=0)
    assert not set(groups[is_val]) & set(groups[~is_val])
    assert 0.18 < is_val.mean() < 0.23


def test_tune_xgb_returns_best_with_early_stopped_trees():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = 3 * X[:, 0] + rng.normal(size=300)
    out = models.tune_xgb(X, y, groups=None, seed=0, n_trials=2)
    assert len(out["trials"]) == 2
    assert out["best"]["val_rmse"] == min(t["val_rmse"] for t in out["trials"])
    assert 1 <= out["best"]["n_estimators"] <= models.TUNE_MAX_TREES


def test_nearest_neighbour_returns_median_of_closest_composition():
    train = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pred = models.nearest_neighbour(train, [10, 30, 90], ["a", "a", "b"], [[0.9, 0.1], [0.1, 0.9]])
    np.testing.assert_allclose(pred, [20.0, 90.0])


def test_hamidieh_settings_match_the_paper():
    m = models.hamidieh_xgb(seed=3)
    assert (m.learning_rate, m.max_depth, m.n_estimators) == (0.02, 16, 374)
    assert (m.subsample, m.colsample_bytree, m.min_child_weight) == (0.5, 0.5, 1)
