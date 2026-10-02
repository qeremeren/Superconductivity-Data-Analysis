"""Phase 3 learning curves, TabPFN side: one billed request per (feature set, split, size).

Grouped splits benchmark.LC_SPLITS, training sizes benchmark.LC_SIZES (the same nested
subsets as the XGBoost side, from benchmark.learning_curve_rows), the split's full test set.
The full-size point is Phase 2's grouped result. Uses the Phase 2 driver: no API call
without --live, cached per-row summaries committed under results/03_learning_curves/tabpfn/,
budget line p3_learning_curves.

Run from the repo root: `uv run python -m experiments.03_tabpfn_learning_curves [--live]`.
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np

from src import benchmark, config, data, models, splits

driver = importlib.import_module("experiments.02_tabpfn")
CACHE = models.TabPFNCache(benchmark.config.RESULTS_DIR / "03_learning_curves" / "tabpfn")


def build_jobs(inputs, feature_sets=data.FEATURE_SETS) -> list[models.TabPFNJob]:
    masks, seeds = splits.load("grouped"), splits.seeds("grouped")
    jobs = []
    for split in benchmark.LC_SPLITS:
        is_test = masks[split]
        for n in benchmark.LC_SIZES:
            rows = benchmark.learning_curve_rows(is_test, split, n)
            for fs in feature_sets:
                X = inputs.features[fs]
                jobs.append(
                    models.TabPFNJob(
                        name=f"lc/{fs}/split_{split:02d}_n{n:05d}",
                        seed=int(seeds[split]),
                        X_train=X.iloc[rows].reset_index(drop=True),
                        y_train=inputs.y[rows],
                        X_test=X[is_test].reset_index(drop=True),
                        y_test=inputs.y[is_test],
                        test_rows=np.flatnonzero(is_test),
                    )
                )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = build_jobs(benchmark.load_inputs())
    out = driver.run(
        jobs,
        config.live_requested(args.live),
        args.workers,
        cache=CACHE,
        budget_lines={"lc": "p3_learning_curves"},
    )
    if out["failed"] or out.get("not_sent"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
