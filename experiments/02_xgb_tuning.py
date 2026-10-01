"""Phase 2 tuned XGBoost: nested random search on each split's training rows only.

For every (split kind, split, feature set), up to MAX_TRIALS configurations are
scored on an inner validation set that mirrors the outer split type (random rows,
or whole composition groups), with early stopping. All trials are saved to
results/02_benchmark/xgb_tuning/<kind>/<features>/split_XX.json. The model is the
best of the first --trials trials (trials are sampled in a fixed seeded order and
are independent, so this equals running only that many), refit on all training
rows, and its test predictions go to predictions/xgb_tuned/<kind>/<features>.parquet.

`--time-first` tunes only random split 0 (both feature sets) with MAX_TRIALS and
prints the projected total and the largest trial count that fits TIME_BUDGET_H.
Run from the repo root: `uv run python -m experiments.02_xgb_tuning [--trials N]`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from src import benchmark, data, models

MAX_TRIALS = 30
TIME_BUDGET_H = 3.0
TUNING_DIR = benchmark.OUT / "xgb_tuning"
WORK_DIR = benchmark.config.RESULTS_DIR / "cache" / "xgb_tuned_predictions"


def tuning_path(kind, features, split):
    return TUNING_DIR / kind / features / f"split_{split:02d}.json"


def tune(kind, split, seed, features, is_test, inputs, n_trials):
    path = tuning_path(kind, features, split)
    if path.is_file():
        record = json.loads(path.read_text())
        if len(record["trials"]) >= n_trials:
            return record
    X = inputs.features[features].to_numpy(np.float32)
    groups = benchmark.inner_groups(kind, inputs)
    start = time.perf_counter()
    out = models.tune_xgb(
        X[~is_test],
        inputs.y[~is_test],
        None if groups is None else groups[~is_test],
        seed,
        n_trials,
    )
    record = {
        "kind": kind,
        "split": split,
        "seed": seed,
        "features": features,
        "seconds": time.perf_counter() - start,
        **out,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return record


def refit_predict(record, n_trials, is_test, inputs):
    best = min(record["trials"][:n_trials], key=lambda t: t["val_rmse"])
    X = inputs.features[record["features"]].to_numpy(np.float32)
    model = models.tuned_xgb(best, record["seed"]).fit(X[~is_test], inputs.y[~is_test])
    return model.predict(X[is_test])


def time_first(inputs):
    per_job = []
    for kind, split, seed, is_test in benchmark.iter_splits(["random"]):
        for fs in data.FEATURE_SETS:
            record = tune(kind, split, seed, fs, is_test, inputs, MAX_TRIALS)
            t = time.perf_counter()
            refit_predict(record, MAX_TRIALS, is_test, inputs)
            per_job.append((record["seconds"] / len(record["trials"]), time.perf_counter() - t))
            print(
                f"{kind} split {split} {fs}: {record['seconds']:.0f}s for "
                f"{len(record['trials'])} trials, refit {per_job[-1][1]:.1f}s"
            )
        break
    n_jobs = sum(len(benchmark.splits.load(k)) for k in benchmark.SPLIT_KINDS) * len(
        data.FEATURE_SETS
    )
    per_trial = float(np.mean([p for p, _ in per_job]))
    refit = float(np.mean([r for _, r in per_job]))
    projected_h = n_jobs * (MAX_TRIALS * per_trial + refit) / 3600
    fit_trials = min(MAX_TRIALS, math.floor((TIME_BUDGET_H * 3600 / n_jobs - refit) / per_trial))
    print(
        json.dumps(
            {
                "jobs": n_jobs,
                "seconds_per_trial": round(per_trial, 2),
                "refit_seconds": round(refit, 2),
                f"projected_hours_at_{MAX_TRIALS}_trials": round(projected_h, 2),
                f"max_trials_within_{TIME_BUDGET_H}h": fit_trials,
            },
            indent=1,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=MAX_TRIALS)
    parser.add_argument("--time-first", action="store_true")
    args = parser.parse_args()
    inputs = benchmark.load_inputs()
    if args.time_first:
        time_first(inputs)
        return

    start, rows = time.perf_counter(), defaultdict(list)
    for kind, split, seed, is_test in benchmark.iter_splits():
        for fs in data.FEATURE_SETS:
            work = WORK_DIR / kind / fs / f"split_{split:02d}_{args.trials}trials.parquet"
            if not work.is_file():
                record = tune(kind, split, seed, fs, is_test, inputs, args.trials)
                pred = refit_predict(record, args.trials, is_test, inputs)
                work.parent.mkdir(parents=True, exist_ok=True)
                benchmark.prediction_frame(split, np.flatnonzero(is_test), pred).to_parquet(work)
            rows[kind, fs].append(pd.read_parquet(work))
        print(f"{kind} split {split:2d} done ({time.perf_counter() - start:.0f}s)", flush=True)
    for (kind, fs), frames in rows.items():
        benchmark.write_predictions(frames, "xgb_tuned", kind, fs)
    print(f"Done in {(time.perf_counter() - start) / 3600:.2f} h with {args.trials} trials")


if __name__ == "__main__":
    main()
