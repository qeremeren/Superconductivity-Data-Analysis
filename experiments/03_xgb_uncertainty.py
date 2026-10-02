"""Phase 3 XGBoost uncertainty baselines (local CPU; no API key, no TabPFN outputs).

On the grouped split (25 splits) and the leave-family-out folds (3), both feature sets,
each starting from that split's Phase 2 tuned parameters (best of the first 26 trials in
results/02_benchmark/xgb_tuning/):

quantile_regression  one multi-quantile XGBoost (objective reg:quantileerror) for the 27
                     levels in metrics.LEVELS_27, fit on all training rows; quantiles are
                     sorted per row to remove crossings.
conformal            split-conformal predictive distribution: the tuned point model is fit
                     on ~80% of the training rows and its signed residuals on the other
                     ~20% (a calibration split drawn with a different seed from the tuning
                     split, grouped like the outer split) are added to the test predictions
                     at each level in LEVELS_27. The calibration rows are never fitted on;
                     the hyperparameters were chosen on an overlapping inner split, which
                     makes the intervals at most slightly optimistic.

Both are summarized like TabPFN (metrics.summarize_distribution) with CRPS on the 20-level
midpoint grid (column crps20). Each job writes, under results/03_uncertainty/xgb/ only:
  <method>/<kind>/<features>/split_SS.parquet and split_SS.json (machine, seconds, settings)
Finished jobs are skipped, so the script resumes after an interruption.

Run from the repo root: `uv run python -m experiments.03_xgb_uncertainty [--threads N]`.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np
from xgboost import XGBRegressor

from src import benchmark, data, metrics, models, splits

OUT = benchmark.config.RESULTS_DIR / "03_uncertainty" / "xgb"
TUNING = benchmark.OUT / "xgb_tuning"
KINDS = ("grouped", "leave_family_out")
METHODS = ("quantile_regression", "conformal")
TRIALS = 26
CALIBRATION_SEED_OFFSET = 7919


def tuned_params(kind, features, split) -> dict:
    record = json.loads((TUNING / kind / features / f"split_{split:02d}.json").read_text())
    return min(record["trials"][:TRIALS], key=lambda t: t["val_rmse"])


def quantile_regression(params, seed, X_train, y_train, X_test, threads):
    best = {k: v for k, v in params.items() if k != "val_rmse"}
    model = XGBRegressor(
        **best,
        objective="reg:quantileerror",
        quantile_alpha=np.asarray(metrics.LEVELS_27),
        tree_method="hist",
        random_state=int(seed),
        n_jobs=threads,
    )
    model.fit(X_train, y_train)
    q = np.sort(model.predict(X_test), axis=1)  # rearrangement removes quantile crossing
    mean = q[:, np.isin(metrics.LEVELS_27, metrics.MIDPOINT_20)].mean(axis=1)  # midpoint rule
    return mean, q


def conformal(params, seed, X_train, y_train, groups, X_test, threads):
    is_cal = models.inner_validation_mask(len(y_train), groups, seed + CALIBRATION_SEED_OFFSET)
    model = models.tuned_xgb(params, seed, n_jobs=threads)
    model.fit(X_train[~is_cal], y_train[~is_cal])
    residuals = y_train[is_cal] - model.predict(X_train[is_cal])
    offsets = np.quantile(residuals, metrics.LEVELS_27, method="inverted_cdf")
    pred = model.predict(X_test)
    return pred, pred[:, None] + offsets[None, :], int(is_cal.sum())


def run_job(method, kind, features, split, seed, is_test, inputs, threads):
    path = OUT / method / kind / features / f"split_{split:02d}.parquet"
    record_path = path.with_suffix(".json")
    if path.is_file() and record_path.is_file():
        return False
    X = inputs.features[features].to_numpy(np.float32)
    y = inputs.y
    params = tuned_params(kind, features, split)
    start = time.perf_counter()
    record = {
        "method": method,
        "kind": kind,
        "features": features,
        "split": split,
        "seed": seed,
        "tuned_params": params,
        "levels": list(metrics.LEVELS_27),
    }
    if method == "quantile_regression":
        mean, q = quantile_regression(params, seed, X[~is_test], y[~is_test], X[is_test], threads)
    else:
        groups = benchmark.inner_groups(kind, inputs)
        mean, q, n_cal = conformal(
            params,
            seed,
            X[~is_test],
            y[~is_test],
            None if groups is None else groups[~is_test],
            X[is_test],
            threads,
        )
        record["calibration_rows"] = n_cal
    summary = metrics.summarize_distribution(
        mean, q, metrics.LEVELS_27, y[is_test], midpoints=metrics.MIDPOINT_20
    )
    summary = summary.rename(columns={"crps": "crps20"})
    summary.insert(0, "y", y[is_test].astype(np.float32))
    summary.insert(0, "row", np.flatnonzero(is_test).astype(np.int32))
    record |= {
        "seconds": round(time.perf_counter() - start, 2),
        "threads": threads,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": benchmark.machine_info(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(path, compression="zstd", index=False)
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=-1, help="XGBoost threads (-1 = all)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    parser.add_argument("--features", nargs="+", default=list(data.FEATURE_SETS))
    parser.add_argument("--splits", nargs="+", type=int, default=None)
    args = parser.parse_args()

    inputs = benchmark.load_inputs()
    todo = []
    for kind in args.kinds:
        masks, seeds = splits.load(kind), splits.seeds(kind)
        for s in range(len(masks)):
            if args.splits is None or s in args.splits:
                todo += [
                    (m, kind, fs, s, int(seeds[s]), masks[s])
                    for fs in args.features
                    for m in args.methods
                ]
    print(f"{len(todo)} jobs on {benchmark.machine_info()['host']}", flush=True)
    start = time.perf_counter()
    for i, (method, kind, fs, s, seed, is_test) in enumerate(todo, 1):
        ran = run_job(method, kind, fs, s, seed, is_test, inputs, args.threads)
        print(
            f"[{i}/{len(todo)}] {method} {kind} {fs} split {s} "
            f"{'done' if ran else 'cached'} ({time.perf_counter() - start:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
