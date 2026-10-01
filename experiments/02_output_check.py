"""Phase 2 output check: 4 billed requests at full benchmark size, before the main runs.

One fit on random split 0's training rows (engineered features), four predicts on
its 7,088 test rows:
  1. output_type="main" with the 107-level grid (metrics.QUANTILE_LEVELS)
  2. output_type="mean", to check that "main" returns the same mean
  3. output_type="main" with a 199-level midpoint grid
  4. output_type="main" with a 999-level midpoint grid (where does the server stop?)
Raw outputs go to results/cache/output_check/ (local) before any analysis; the
report goes to results/02_benchmark/output_check.json. --from-saved redoes the
analysis without requests.

Run from the repo root: `uv run python -m experiments.02_output_check --live`.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np

from src import benchmark, budget, config, metrics, splits

OUT = benchmark.OUT / "output_check.json"
RAW = config.RESULTS_DIR / "cache" / "output_check"
GRIDS = {
    "main_107": metrics.QUANTILE_LEVELS,
    "main_199": np.round((np.arange(199) + 0.5) / 199, 6),
    "main_999": np.round((np.arange(999) + 0.5) / 999, 6),
}


def requests(X_train, y_train, X_test, seed, run_budget):
    from tabpfn_client import TabPFNRegressor

    reg = TabPFNRegressor(model_path=config.TABPFN_MODEL_PATH, random_state=seed)
    reg.fit(X_train, y_train)
    report = {}
    order = [("main_107", "main"), ("mean", "mean"), ("main_199", "main"), ("main_999", "main")]
    for name, output_type in order:
        kwargs = {"quantiles": [float(x) for x in GRIDS[name]]} if name in GRIDS else {}
        run_budget.charge(1, check=name, output_type=output_type)
        t = time.perf_counter()
        try:
            out = reg.predict(X_test, output_type=output_type, **kwargs)
        except Exception as exc:
            report[name] = {"ok": False, "error": repr(exc)[:500]}
            traceback.print_exc()
            continue
        seconds = time.perf_counter() - t
        if output_type == "mean":
            arrays = {"mean": np.asarray(out, float)}
        else:
            arrays = {
                "mean": np.asarray(out["mean"], float),
                "median": np.asarray(out["median"], float),
                "quantiles": np.asarray(out["quantiles"], float),
            }
            arrays["keys"] = np.array(sorted(out))
        np.savez_compressed(RAW / f"{name}.npz", **arrays)
        report[name] = {
            "ok": True,
            "seconds": round(seconds, 1),
            "meta": dict(reg._last_meta),
            "timings": reg.get_timings(),
        }
    return report


def analyze(report, y_test):
    for name, entry in report.items():
        if not isinstance(entry, dict) or not entry.get("ok"):
            continue
        with np.load(RAW / f"{name}.npz") as f:
            arrays = {k: f[k] for k in f.files}
        entry["arrays"] = {k: list(v.shape) for k, v in arrays.items() if k != "keys"}
        if "keys" in arrays:
            entry["keys"] = arrays["keys"].tolist()
        if name in GRIDS:
            levels = GRIDS[name]
            q = arrays["quantiles"]
            q = q.T if q.shape == (len(levels), len(y_test)) else q
            entry["quantile_matrix_rows_by_levels"] = list(q.shape)
            entry["monotone_rows_share"] = float(np.mean(np.all(np.diff(q, axis=1) >= -1e-9, 1)))
            entry["rmse_of_mean_K"] = float(np.sqrt(np.mean((arrays["mean"] - y_test) ** 2)))
    if report.get("main_107", {}).get("ok") and report.get("mean", {}).get("ok"):
        with np.load(RAW / "main_107.npz") as a, np.load(RAW / "mean.npz") as b:
            report["main_mean_vs_plain_mean_max_abs_K"] = float(
                np.max(np.abs(a["mean"] - b["mean"]))
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--from-saved", action="store_true")
    args = parser.parse_args()
    inputs = benchmark.load_inputs()
    is_test = splits.load("random")[0]
    seed = int(splits.seeds("random")[0])
    X = inputs.features["engineered"]
    X_train, X_test = X[~is_test].reset_index(drop=True), X[is_test].reset_index(drop=True)
    y_train, y_test = inputs.y[~is_test], inputs.y[is_test]

    if args.from_saved:
        report = json.loads(OUT.read_text())["requests"]
    else:
        if not config.live_requested(args.live):
            raise SystemExit("This check makes 4 billed requests; pass --live to run it.")
        tokens = budget.estimate_tokens(X_train, X_test)
        run_budget = budget.authorize("p2_output_check", 4, tokens)
        RAW.mkdir(parents=True, exist_ok=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        report = requests(X_train, y_train, X_test, seed, run_budget)
        OUT.write_text(json.dumps({"requests": report}, indent=1, default=str) + "\n")
    report = analyze(report, y_test)
    OUT.write_text(
        json.dumps(
            {"n_train": len(X_train), "n_test": len(X_test), "seed": seed, "requests": report},
            indent=1,
            default=str,
        )
        + "\n"
    )
    print(json.dumps(report, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
