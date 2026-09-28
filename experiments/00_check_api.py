"""Phase 0 API check: token works, pinned model resolves, and what things cost.

Default mode only calls estimate_cost(), which sends dataset dimensions and
consumes no quota. --live (or TABPFN_LIVE=1) adds one tiny fit/predict on 200
training rows (~10k tokens, the per-request minimum) to record the server-side
model version and the shape of output_type="full".

Run from the repo root: `make check-api` (add LIVE=1 for the live call), or
`uv run python -m experiments.00_check_api [--live]`. Writes results/00_api_check.json.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime

import numpy as np
import tabpfn_client
from tabpfn_client import TabPFNRegressor, estimate_cost, get_api_usage

from src import config, data

OUT = config.RESULTS_DIR / "00_api_check.json"

# The paper's protocol: random 2/3 train / 1/3 test on all 21,263 rows.
N_TRAIN_RANDOM = round(data.N_ROWS * 2 / 3)
N_TEST_RANDOM = data.N_ROWS - N_TRAIN_RANDOM


def cost_scenarios(engineered, composition):
    """(name, X_train, X_test) at the sizes later phases will use.

    Only shapes reach the server. The discovery pool is an upper bound; the
    real pool is the deduplicated formula set from Phase 1.
    """
    return [
        ("random_split_engineered", engineered[:N_TRAIN_RANDOM], engineered[N_TRAIN_RANDOM:]),
        ("random_split_composition", composition[:N_TRAIN_RANDOM], composition[N_TRAIN_RANDOM:]),
        ("learning_curve_100_engineered", engineered[:100], engineered[N_TRAIN_RANDOM:]),
        ("discovery_round_100_labeled", composition[:100], composition[100:]),
        ("live_smoke_check", engineered[:200], engineered[200:300]),
    ]


def describe(value):
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, list):
        return {"type": "list", "len": len(value), "items": [describe(v) for v in value[:3]]}
    return {"type": type(value).__name__}


def live_smoke_check(engineered, target):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(engineered), size=300, replace=False)
    train_idx, test_idx = idx[:200], idx[200:]

    reg = TabPFNRegressor(model_path=config.TABPFN_MODEL_PATH, random_state=0)
    reg.fit(engineered.iloc[train_idx], target.iloc[train_idx])
    out = reg.predict(engineered.iloc[test_idx], output_type="full")

    mean = np.asarray(out["mean"])
    borders = np.asarray(out["borders"])
    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "last_meta": reg.last_meta,
        "timings": reg.get_timings(),
        "full_output": {k: describe(v) for k, v in out.items()},
        "n_buckets": int(np.asarray(out["logits"]).shape[1]),
        "borders_monotonic": bool(np.all(np.diff(borders) > 0)),
        "borders_range_K": [float(borders.min()), float(borders.max())],
        "mean_all_finite": bool(np.isfinite(mean).all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="also run one tiny fit/predict")
    args = parser.parse_args()
    live = config.live_requested(args.live)
    config.require_tabpfn_token()

    train = data.load_train()
    unique_m = data.load_unique_m()
    engineered = train.drop(columns=[data.TARGET])
    composition = unique_m.iloc[:, : data.N_ELEMENTS]
    target = train[data.TARGET]

    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "tabpfn_client": tabpfn_client.__version__,
        "model_path": config.TABPFN_MODEL_PATH,
        "model_version": config.TABPFN_MODEL_VERSION,
        "client_known_models": TabPFNRegressor.list_available_models(),
        "cost_estimates": {},
    }

    for name, X_train, X_test in cost_scenarios(engineered, composition):
        quote = estimate_cost(X_train, X_test, model_version=config.TABPFN_MODEL_VERSION)
        report["cost_estimates"][name] = quote.model_dump(mode="json")
        print(f"{name:32s} {X_train.shape} -> {len(X_test)} rows: {quote.estimated_cost:,} tokens")

    if live:
        report["usage_before"] = get_api_usage()
        report["live_smoke_check"] = live_smoke_check(engineered, target)
        report["usage_after"] = get_api_usage()
        print(json.dumps(report["live_smoke_check"], indent=2, default=str))
    else:
        print("Skipped live fit/predict (pass --live or set TABPFN_LIVE=1).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"Wrote {OUT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
