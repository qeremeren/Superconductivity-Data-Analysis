"""Phase 3 learning curves, XGBoost side (local CPU; no API key, no TabPFN outputs).

For grouped splits benchmark.LC_SPLITS and training sizes benchmark.LC_SIZES, both
feature sets: published-settings XGBoost, and XGBoost tuned by the same nested search
as Phase 2 (26 trials, inner split grouped by composition) on that size's rows only.
Training rows are the nested subsets from benchmark.learning_curve_rows, identical to
the TabPFN side; the test set is the split's full test set.

Each job writes, under results/03_learning_curves/xgb/ only:
  predictions/<model>/<features>/split_SS_nNNNNN.parquet   (row, pred)
  records/<model>/<features>/split_SS_nNNNNN.json          (machine, seconds, params)
Finished jobs are skipped, so the script resumes after an interruption.

Run from the repo root: `uv run python -m experiments.03_xgb_learning_curves [--threads N]`.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np

from src import benchmark, data, models, splits

OUT = benchmark.config.RESULTS_DIR / "03_learning_curves" / "xgb"
TRIALS = 26
MODELS = ("xgb_hamidieh", "xgb_tuned")


def job_paths(model, features, split, n):
    stem = f"split_{split:02d}_n{n:05d}"
    return (
        OUT / "predictions" / model / features / f"{stem}.parquet",
        OUT / "records" / model / features / f"{stem}.json",
    )


def run_job(model, features, split, seed, n, is_test, inputs, threads):
    pred_path, record_path = job_paths(model, features, split, n)
    if pred_path.is_file() and record_path.is_file():
        return False
    rows = benchmark.learning_curve_rows(is_test, split, n)
    X = inputs.features[features].to_numpy(np.float32)
    y = inputs.y
    start = time.perf_counter()
    record = {
        "model": model,
        "features": features,
        "kind": "grouped",
        "split": split,
        "seed": seed,
        "n_train": int(len(rows)),
    }
    if model == "xgb_hamidieh":
        fitted = models.hamidieh_xgb(seed, n_jobs=threads).fit(X[rows], y[rows])
    else:
        tuning = models.tune_xgb(
            X[rows], y[rows], inputs.composition_key[rows], seed, TRIALS, n_jobs=threads
        )
        record["tuning"] = tuning
        fitted = models.tuned_xgb(tuning["best"], seed, n_jobs=threads).fit(X[rows], y[rows])
    pred = fitted.predict(X[is_test])
    record |= {
        "seconds": round(time.perf_counter() - start, 2),
        "threads": threads,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": benchmark.machine_info(),
    }
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.prediction_frame(split, np.flatnonzero(is_test), pred).to_parquet(pred_path)
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=-1, help="XGBoost threads (-1 = all)")
    parser.add_argument("--splits", nargs="+", type=int, default=list(benchmark.LC_SPLITS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(benchmark.LC_SIZES))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--features", nargs="+", default=list(data.FEATURE_SETS))
    args = parser.parse_args()

    inputs = benchmark.load_inputs()
    masks, seeds = splits.load("grouped"), splits.seeds("grouped")
    todo = [
        (m, fs, s, n)
        for s in args.splits
        for fs in args.features
        for n in args.sizes
        for m in args.models
    ]
    print(f"{len(todo)} jobs on {benchmark.machine_info()['host']}", flush=True)
    start = time.perf_counter()
    for i, (model, fs, s, n) in enumerate(todo, 1):
        ran = run_job(model, fs, s, int(seeds[s]), n, masks[s], inputs, args.threads)
        print(
            f"[{i}/{len(todo)}] {model} {fs} split {s} n={n} "
            f"{'done' if ran else 'cached'} ({time.perf_counter() - start:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
