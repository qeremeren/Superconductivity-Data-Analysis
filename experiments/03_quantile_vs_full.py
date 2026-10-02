"""Phase 3 check: how accurate are the 107-level quantile-grid summaries?

For grouped splits 0 and 1 (composition features), one billed output_type="full" request
on the first 400 test rows, with the same training rows and seed as the Phase 2 job. The
exact bar-distribution mean, CRPS, PIT, P(Tc > 77 K) and summary quantiles are compared
with Phase 2's committed grid-based summary for the same rows. The means also test that a
row's prediction does not depend on which other rows are in the test set.

Raw logits go to results/cache/ (local) before any analysis; the comparison is written to
results/03_uncertainty/quantile_vs_full.json. --from-saved redoes the analysis only.
Run from the repo root: `uv run python -m experiments.03_quantile_vs_full --live`.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import benchmark, budget, config, metrics, splits

SPLITS = (0, 1)
FEATURES = "composition"
N_ROWS = 400  # the server's per-request cap for output_type="full"
RAW = config.RESULTS_DIR / "cache" / "quantile_vs_full"
OUT = config.RESULTS_DIR / "03_uncertainty" / "quantile_vs_full.json"
PHASE2 = benchmark.OUT / "tabpfn" / "grouped" / FEATURES


def request(inputs, split, seed, is_test, run_budget):
    from tabpfn_client import TabPFNRegressor

    X = inputs.features[FEATURES]
    rows = np.flatnonzero(is_test)[:N_ROWS]
    reg = TabPFNRegressor(model_path=config.TABPFN_MODEL_PATH, random_state=seed)
    reg.fit(X[~is_test].reset_index(drop=True), inputs.y[~is_test])
    run_budget.charge(1, split=split, output_type="full", rows=N_ROWS)
    out = reg.predict(X.iloc[rows].reset_index(drop=True), output_type="full")
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RAW / f"split_{split:02d}.npz",
        rows=rows,
        logits=np.asarray(out["logits"], np.float32),
        borders=np.asarray(out["borders"], np.float64),
        mean=np.asarray(out["mean"], np.float64),
    )


def compare(split, inputs) -> dict:
    with np.load(RAW / f"split_{split:02d}.npz") as f:
        rows, logits, borders, mean = f["rows"], f["logits"].astype(float), f["borders"], f["mean"]
    y = inputs.y[rows]
    grid = pd.read_parquet(PHASE2 / f"split_{split:02d}.parquet").set_index("row").loc[rows]
    exact_q = metrics.bar_quantiles(logits, borders, metrics.SUMMARY_LEVELS)
    exact = {
        "crps": metrics.bar_crps(logits, borders, y),
        "pit": metrics.bar_cdf(logits, borders, y),
        "p_above_77K": 1 - metrics.bar_cdf(logits, borders, 77.0),
    }

    def gap(a, b):
        d = np.abs(np.asarray(a, float) - np.asarray(b, float))
        return {"max": float(d.max()), "median": float(np.median(d))}

    return {
        "rows": int(len(rows)),
        "mean_full_vs_phase2_K": gap(mean, grid["mean"]),
        "crps_exact_vs_grid_K": gap(exact["crps"], grid["crps"]),
        "crps_mean_exact_vs_grid_K": [float(exact["crps"].mean()), float(grid["crps"].mean())],
        "pit_exact_vs_grid": gap(exact["pit"], grid["pit"]),
        "p_above_77K_exact_vs_grid": gap(exact["p_above_77K"], grid["p_above_77K"]),
        "p_above_77K_exact_below_grid_floor": int((exact["p_above_77K"] < 0.005).sum()),
        "quantiles_exact_vs_grid_K": gap(
            exact_q, grid[[f"q{lvl:g}" for lvl in metrics.SUMMARY_LEVELS]].to_numpy()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--from-saved", action="store_true")
    args = parser.parse_args()
    inputs = benchmark.load_inputs()
    masks, seeds = splits.load("grouped"), splits.seeds("grouped")
    if not args.from_saved:
        missing = [s for s in SPLITS if not (RAW / f"split_{s:02d}.npz").is_file()]
        if missing:
            if not config.live_requested(args.live):
                raise SystemExit("Needs one billed request per split; pass --live.")
            X = inputs.features[FEATURES]
            tokens = budget.estimate_tokens(X[~masks[0]], X.iloc[:N_ROWS])
            run_budget = budget.authorize("p3_quantile_vs_full", len(missing), tokens)
            for s in missing:
                request(inputs, s, int(seeds[s]), masks[s], run_budget)
    report = {f"split_{s:02d}": compare(s, inputs) for s in SPLITS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
