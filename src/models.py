"""Models for the benchmarks, and the cache that makes TabPFN results reproducible
without an API key.

Local models: linear regression (standardized on train only), XGBoost with the
published Hamidieh (2018) settings, XGBoost tuned by a nested random search on
each split's training rows, and a 1-nearest-neighbour composition lookup.

TabPFN-3.5 runs through the API. Each job's committed output is a per-row
summary (src/metrics.summarize_distribution) plus a JSON record whose fingerprint
covers the model, client version, quantile grid, seed and input data. The full
quantile grid is kept only locally (results/cache/, gitignored), so a summary can
be recomputed without a new request. TabPFNCache.get() returns a cached summary,
refuses a stale one, and calls the API only when live calls are allowed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src import config, metrics

# Hamidieh (2018), Table 4: the grid-search optimum used for the published RMSE.
HAMIDIEH_XGB = {
    "learning_rate": 0.02,
    "max_depth": 16,
    "min_child_weight": 1,
    "colsample_bytree": 0.5,
    "subsample": 0.5,
    "n_estimators": 374,
    "tree_method": "exact",  # what R's xgboost used on data this small
}

RAW_CACHE = config.RESULTS_DIR / "cache" / "tabpfn_raw"


def linear_regression():
    return make_pipeline(StandardScaler(), LinearRegression())


# Phase 5 discovery baseline, fixed in the Phase 5 plan before any GP run: scikit-learn GP,
# amplitude x Matern 5/2 with one length scale + white noise, normalised Tc, 3 optimizer
# restarts, element-fraction features, sklearn's default hyperparameter bounds.
GP_RESTARTS = 3


def gp_predict(X_train, y_train, X_pool, seed: int) -> dict:
    """Fit the GP baseline and return its predictive mean and SD (noise included) on the pool."""
    import time
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1.0)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=GP_RESTARTS,
        random_state=int(seed),
    )
    t = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(np.asarray(X_train, float), np.asarray(y_train, float))
        mean, sd = gp.predict(np.asarray(X_pool, float), return_std=True)
    n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return {
        "mean": mean,
        "sd": sd,
        "quantiles": None,
        "seconds": time.perf_counter() - t,
        "meta": {
            "model_path": "sklearn GaussianProcessRegressor",
            "fitted": f"{gp.kernel_}" + (f" [{n_warn} convergence warnings]" if n_warn else ""),
        },
    }


def hamidieh_xgb(seed: int, n_jobs: int = -1) -> XGBRegressor:
    return XGBRegressor(**HAMIDIEH_XGB, random_state=int(seed), n_jobs=n_jobs)


# --- nested XGBoost tuning -------------------------------------------------------

TUNE_MAX_TREES = 2000
TUNE_EARLY_STOP = 50
INNER_VAL_FRACTION = 0.2


def sample_xgb_params(rng: np.random.Generator) -> dict:
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.3)))),
        "max_depth": int(rng.integers(3, 17)),
        "min_child_weight": float(np.exp(rng.uniform(0, np.log(32)))),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.3, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.01), np.log(30)))),
    }


def inner_validation_mask(n_rows: int, groups, seed: int) -> np.ndarray:
    """~20% of the training rows for validation; whole groups when groups is given,
    mirroring the outer split type."""
    rng = np.random.default_rng(seed)
    if groups is None:
        mask = np.zeros(n_rows, dtype=bool)
        mask[rng.permutation(n_rows)[: round(INNER_VAL_FRACTION * n_rows)]] = True
        return mask
    labels, inverse, sizes = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    order = rng.permutation(len(labels))
    n_val = int(np.searchsorted(np.cumsum(sizes[order]), INNER_VAL_FRACTION * n_rows)) + 1
    is_val = np.zeros(len(labels), dtype=bool)
    is_val[order[:n_val]] = True
    return is_val[inverse]


def tune_xgb(X, y, groups, seed: int, n_trials: int, n_jobs: int = -1) -> dict:
    """Random search on one inner train/validation split of the training rows only.

    Returns the best parameters with n_estimators set by early stopping, and every
    trial's validation RMSE.
    """
    X, y = np.asarray(X, np.float32), np.asarray(y, float)
    is_val = inner_validation_mask(len(y), groups, seed)
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_trials):
        params = sample_xgb_params(rng)
        model = XGBRegressor(
            **params,
            n_estimators=TUNE_MAX_TREES,
            early_stopping_rounds=TUNE_EARLY_STOP,
            tree_method="hist",
            random_state=int(seed),
            n_jobs=n_jobs,
        )
        model.fit(X[~is_val], y[~is_val], eval_set=[(X[is_val], y[is_val])], verbose=False)
        trials.append(
            {
                **params,
                "n_estimators": int(model.best_iteration) + 1,
                "val_rmse": float(model.best_score),
            }
        )
    best = min(trials, key=lambda t: t["val_rmse"])
    return {"best": best, "trials": trials, "inner_val_rows": int(is_val.sum())}


def tuned_xgb(best: dict, seed: int, n_jobs: int = -1, **overrides) -> XGBRegressor:
    params = {k: v for k, v in best.items() if k != "val_rmse"} | overrides
    return XGBRegressor(**params, tree_method="hist", random_state=int(seed), n_jobs=n_jobs)


# --- 1-nearest-neighbour composition lookup ---------------------------------------


def nearest_neighbour(train_fractions, train_y, train_keys, test_fractions, chunk=1000):
    """Predict each test row by the median Tc of its nearest training composition
    (L1 over element fractions): a pure memorization reference."""
    train_y = pd.Series(np.asarray(train_y, float))
    per_key = train_y.groupby(np.asarray(train_keys)).median()
    first = pd.Series(np.arange(len(train_y))).groupby(np.asarray(train_keys)).first()
    vecs = np.asarray(train_fractions)[first.loc[per_key.index].to_numpy()]
    test_fractions = np.asarray(test_fractions)
    pred = np.empty(len(test_fractions))
    for start in range(0, len(test_fractions), chunk):
        d = cdist(test_fractions[start : start + chunk], vecs, metric="cityblock")
        pred[start : start + chunk] = per_key.to_numpy()[d.argmin(axis=1)]
    return pred


# --- TabPFN through the API -------------------------------------------------------


def tabpfn_predict(X_train, y_train, X_test, seed: int, levels=metrics.QUANTILE_LEVELS) -> dict:
    """One fit (free) and one billed predict request: mean plus the quantile grid."""
    from tabpfn_client import TabPFNRegressor

    reg = TabPFNRegressor(model_path=config.TABPFN_MODEL_PATH, random_state=int(seed))
    reg.fit(X_train, y_train)
    out = reg.predict(X_test, output_type="main", quantiles=[float(x) for x in levels])
    q = np.asarray(out["quantiles"], dtype=np.float64)
    if q.shape == (len(levels), len(X_test)):
        q = q.T
    if q.shape != (len(X_test), len(levels)):
        raise ValueError(f"unexpected quantile output shape {q.shape}")
    return {
        "mean": np.asarray(out["mean"], dtype=np.float64),
        "quantiles": q,
        # The docs FAQ calls this `last_meta`; tabpfn-client 0.6.1 only has `_last_meta`.
        "meta": dict(reg._last_meta),
        "timings": reg.get_timings(),
    }


def fingerprint(X_train, y_train, X_test, seed: int, levels=metrics.QUANTILE_LEVELS) -> str:
    h = hashlib.sha256()
    for part in (
        config.TABPFN_MODEL_PATH,
        version("tabpfn-client"),
        str(int(seed)),
        np.asarray(levels, np.float64).tobytes(),
        ",".join(map(str, X_train.columns)),
        np.ascontiguousarray(X_train.to_numpy(np.float64)).tobytes(),
        np.ascontiguousarray(np.asarray(y_train, np.float64)).tobytes(),
        np.ascontiguousarray(X_test.to_numpy(np.float64)).tobytes(),
    ):
        h.update(part if isinstance(part, bytes) else part.encode())
    return h.hexdigest()


class StaleCache(RuntimeError):
    pass


class MissingCache(RuntimeError):
    pass


@dataclass(frozen=True)
class TabPFNJob:
    name: str  # relative path, e.g. "grouped/composition/split_03"
    seed: int
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    test_rows: np.ndarray


@dataclass(frozen=True)
class TabPFNCache:
    """Committed summaries under out_dir; raw grids under RAW_CACHE (local only)."""

    out_dir: Path

    def summary_path(self, job: TabPFNJob) -> Path:
        return self.out_dir / f"{job.name}.parquet"

    def record_path(self, job: TabPFNJob) -> Path:
        return self.out_dir / f"{job.name}.json"

    def raw_path(self, job: TabPFNJob) -> Path:
        return RAW_CACHE / self.out_dir.name / f"{job.name}.npz"

    def load(self, job: TabPFNJob) -> pd.DataFrame | None:
        """The cached summary, None if absent; raises StaleCache on a fingerprint mismatch.
        A raw grid without a summary is summarized again, with no request."""
        fp = fingerprint(job.X_train, job.y_train, job.X_test, job.seed)
        if self.summary_path(job).is_file():
            record = json.loads(self.record_path(job).read_text())
            if record["fingerprint"] != fp:
                raise StaleCache(f"{job.name}: cached summary was made from different inputs")
            return pd.read_parquet(self.summary_path(job))
        if self.raw_path(job).is_file():
            with np.load(self.raw_path(job), allow_pickle=False) as raw:
                if str(raw["fingerprint"]) != fp:
                    raise StaleCache(f"{job.name}: raw grid was made from different inputs")
                record = json.loads(str(raw["record"]))
                quantiles = raw["quantiles"] if "quantiles" in raw.files else None
                self._write_summary(job, raw["mean"], quantiles, record)
            return pd.read_parquet(self.summary_path(job))
        return None

    def save_raw(self, job: TabPFNJob, result: dict) -> None:
        """Called straight after the request returns, before any analysis."""
        fp = fingerprint(job.X_train, job.y_train, job.X_test, job.seed)
        record = {
            "fingerprint": fp,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model_path": config.TABPFN_MODEL_PATH,
            "tabpfn_client": version("tabpfn-client"),
            "seed": int(job.seed),
            "n_train": len(job.X_train),
            "n_test": len(job.X_test),
            "server_meta": result["meta"],
            "timings": result["timings"],
        }
        path = self.raw_path(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"mean": result["mean"], "levels": np.asarray(metrics.QUANTILE_LEVELS)}
        if result.get("quantiles") is not None:  # None for mean-only output (Thinking)
            arrays["quantiles"] = result["quantiles"].astype(np.float32)
        np.savez_compressed(
            path,
            **arrays,
            fingerprint=np.str_(fp),
            record=np.str_(json.dumps(record, default=str)),
        )

    def _write_summary(self, job: TabPFNJob, mean, quantiles, record: dict) -> None:
        if quantiles is None:
            summary = pd.DataFrame({"mean": np.asarray(mean, np.float32)})
        else:
            summary = metrics.summarize_distribution(
                mean, np.asarray(quantiles, float), metrics.QUANTILE_LEVELS, job.y_test
            )
        summary.insert(0, "y", np.asarray(job.y_test, np.float32))
        summary.insert(0, "row", np.asarray(job.test_rows, np.int32))
        path = self.summary_path(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(path, compression="zstd", index=False)
        self.record_path(job).write_text(json.dumps(record, indent=1, default=str) + "\n")

    def get(self, job: TabPFNJob, live: bool, budget=None, predict=tabpfn_predict) -> pd.DataFrame:
        cached = self.load(job)
        if cached is not None:
            return cached
        if not live:
            raise MissingCache(
                f"{job.name}: no cached TabPFN output. Reproducing never calls the API; "
                "to make the request, rerun with --live (needs TABPFN_TOKEN)."
            )
        if budget is None:
            raise ValueError("a live TabPFN call needs a RunBudget from budget.authorize()")
        budget.charge(1, job=job.name, output_type="main")
        result = predict(job.X_train, job.y_train, job.X_test, job.seed)
        self.save_raw(job, result)
        return self.load(job)
