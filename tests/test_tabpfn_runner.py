"""The Phase 2 TabPFN driver: reproduce mode never calls the API; live mode charges
one request per missing job and resumes from the cache."""

import functools
import importlib

import numpy as np
import pytest
from scipy import stats

from src import benchmark, budget, data, metrics, models

runner = importlib.import_module("experiments.02_tabpfn")
needs_data = pytest.mark.skipif(
    not data.is_downloaded(), reason="data/raw/ missing or unverified; run `make data`"
)
USAGE = {
    "monthly_token_limit": 20_000_000,
    "monthly_tokens_used": 0,
    "daily_token_limit": 15_000_000,
    "daily_tokens_used": 0,
}


def fake_predict(X_train, y_train, X_test, seed):
    q = np.tile(stats.norm.ppf(metrics.QUANTILE_LEVELS, 40, 10), (len(X_test), 1))
    return {"mean": np.full(len(X_test), 40.0), "quantiles": q, "meta": {}, "timings": {}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "RAW_CACHE", tmp_path / "raw")
    ledger = tmp_path / "ledger.jsonl"
    return {
        "cache": models.TabPFNCache(tmp_path / "out"),
        "authorize": functools.partial(budget.authorize, usage=USAGE, ledger=ledger),
        "estimate": lambda X_train, X_test: 10_000,
        "ledger": ledger,
    }


@needs_data
def test_reproduce_mode_refuses_and_live_mode_resumes(env):
    jobs = runner.build_jobs(benchmark.load_inputs(), ["leave_family_out"], ["composition"])
    kw = {k: env[k] for k in ("cache", "authorize", "estimate")}
    with pytest.raises(models.MissingCache):
        runner.run(jobs, live=False, workers=1, predict=fake_predict, **kw)
    assert budget.read_ledger(env["ledger"]) == []

    first = runner.run(jobs, live=True, workers=1, predict=fake_predict, **kw)
    again = runner.run(jobs, live=True, workers=1, predict=fake_predict, **kw)
    assert first["requested"] == 3 and again["requested"] == 0
    assert sum(e["requests"] for e in budget.read_ledger(env["ledger"])) == 3
    assert runner.run(jobs, live=False, workers=1, predict=fake_predict, **kw)["requested"] == 0
