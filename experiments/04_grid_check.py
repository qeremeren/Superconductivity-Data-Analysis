"""Phase 4 check: the tail-dense grid (146 levels) vs a uniform 999-level grid.

Main scenario, pilot seed 100's start set (50 materials): one TabPFN fit, two billed
predict requests on the whole unlabeled pool, one per grid. Compares the acquisition
quantities the loop uses (EI over y* = min(best, threshold), q90, P(top 1%)) per material,
the rank agreement, and which materials each grid would select first; also times both.
Raw outputs go to results/cache/ first; the report to results/04_discovery/grid_check.json.

Run from the repo root: `uv run python -m experiments.04_grid_check --live [--from-saved]`.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import spearmanr

from src import budget, config, discovery, metrics

SEED = discovery.PILOT_SEEDS[0]
GRID_999 = np.round((np.arange(999) + 0.5) / 999, 6)
RAW = config.RESULTS_DIR / "cache" / "grid_check"
OUT = config.RESULTS_DIR / "04_discovery" / "grid_check.json"


def requests(scn, lab, unl):
    from tabpfn_client import TabPFNRegressor

    X_train, X_pool = scn.X.iloc[lab].reset_index(drop=True), scn.X.iloc[unl].reset_index(drop=True)
    tokens = budget.estimate_tokens(X_train, X_pool)
    run_budget = budget.authorize("p4_grid_check", 2, tokens)
    reg = TabPFNRegressor(model_path=config.TABPFN_MODEL_PATH, random_state=SEED)
    reg.fit(X_train, scn.tc[lab])
    RAW.mkdir(parents=True, exist_ok=True)
    for name, levels in (("tail", discovery.TAIL_LEVELS), ("u999", GRID_999)):
        run_budget.charge(1, grid=name)
        t = time.perf_counter()
        out = reg.predict(X_pool, output_type="main", quantiles=[float(x) for x in levels])
        seconds = time.perf_counter() - t
        q = np.asarray(out["quantiles"], np.float32)
        q = q.T if q.shape[0] == len(levels) else q
        np.savez_compressed(
            RAW / f"{name}.npz",
            mean=np.asarray(out["mean"]),
            quantiles=q,
            levels=levels,
            seconds=seconds,
            pool_index=unl,
        )


def scores(name, scn, y_star):
    with np.load(RAW / f"{name}.npz") as f:
        mean, q, levels, seconds = (
            f["mean"],
            f["quantiles"].astype(float),
            f["levels"],
            f["seconds"],
        )
    if name == "tail":
        weights = discovery.TAIL_WEIGHTS
        q90 = q[:, np.isclose(levels, 0.9)][:, 0]
    else:
        weights = np.full(len(levels), 1 / len(levels))
        q90 = np.array([np.interp(0.9, levels, row) for row in q])
    return {
        "ei": discovery.expected_improvement(q, weights, y_star),
        "q90": q90,
        "p_top1": 1 - metrics.cdf_at(q, levels, scn.threshold),
        "mean": mean,
        "seconds": float(seconds),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--from-saved", action="store_true")
    args = parser.parse_args()
    scn = discovery.load_scenario("main")
    lab = discovery.initial_set(scn, SEED)
    unl = np.setdiff1d(np.arange(len(scn.keys)), lab)
    y_star = min(float(scn.tc[lab].max()), scn.threshold)
    if not args.from_saved:
        if not config.live_requested(args.live):
            raise SystemExit("Two billed requests; pass --live.")
        requests(scn, lab, unl)
    a, b = scores("tail", scn, y_star), scores("u999", scn, y_star)
    report = {
        "pool": len(unl),
        "train": len(lab),
        "y_star_K": y_star,
        "threshold_K": scn.threshold,
        "grid_levels": {"tail": len(discovery.TAIL_LEVELS), "u999": len(GRID_999)},
        "request_seconds": {"tail": a["seconds"], "u999": b["seconds"]},
        "mean_max_abs_diff_K": float(np.abs(a["mean"] - b["mean"]).max()),
    }
    for k in ("ei", "q90", "p_top1"):
        d = np.abs(a[k] - b[k])
        top_a = set(np.argsort(-a[k])[: discovery.BATCH])
        top_b = set(np.argsort(-b[k])[: discovery.BATCH])
        report[k] = {
            "max_abs_diff": float(d.max()),
            "median_abs_diff": float(np.median(d)),
            "mean_tail": float(a[k].mean()),
            "mean_999": float(b[k].mean()),
            "spearman": float(spearmanr(a[k], b[k]).statistic),
            "same_first_batch": len(top_a & top_b),
            "batch": discovery.BATCH,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
