"""Phase 2 local baselines on every split (no API calls).

linear        standardized linear regression (scaler fit on train only)
xgb_hamidieh  XGBoost with the published Hamidieh (2018) settings
nn_lookup     1-nearest-neighbour composition lookup (composition features only)

Writes results/02_benchmark/predictions/<model>/<split kind>/<features>.parquet.
Run from the repo root: `uv run python -m experiments.02_local_models [--models ...]`.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from src import benchmark, data, models

MODELS = ("linear", "xgb_hamidieh", "nn_lookup")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--kinds", nargs="+", default=list(benchmark.SPLIT_KINDS))
    args = parser.parse_args()

    inputs = benchmark.load_inputs()
    rows = defaultdict(list)  # (model, kind, features) -> prediction frames
    start = time.perf_counter()
    for kind, i, seed, is_test in benchmark.iter_splits(args.kinds):
        test_rows = np.flatnonzero(is_test)
        y_train = inputs.y[~is_test]
        for fs in data.FEATURE_SETS:
            X = inputs.features[fs].to_numpy(np.float32)
            if "linear" in args.models:
                pred = models.linear_regression().fit(X[~is_test], y_train).predict(X[is_test])
                rows["linear", kind, fs].append(benchmark.prediction_frame(i, test_rows, pred))
            if "xgb_hamidieh" in args.models:
                model = models.hamidieh_xgb(seed).fit(X[~is_test], y_train)
                pred = model.predict(X[is_test])
                rows["xgb_hamidieh", kind, fs].append(
                    benchmark.prediction_frame(i, test_rows, pred)
                )
        if "nn_lookup" in args.models:
            fr = inputs.features["composition"].to_numpy()
            pred = models.nearest_neighbour(
                fr[~is_test], y_train, inputs.composition_key[~is_test], fr[is_test]
            )
            rows["nn_lookup", kind, "composition"].append(
                benchmark.prediction_frame(i, test_rows, pred)
            )
        print(f"{kind} split {i:2d} done ({time.perf_counter() - start:.0f}s)", flush=True)

    for (model, kind, fs), frames in rows.items():
        benchmark.write_predictions(frames, model, kind, fs)
    print(f"Wrote {len(rows)} prediction files in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
