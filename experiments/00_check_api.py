"""Phase 0 API check: token works, pinned model resolves, and what things cost.

Default mode only calls estimate_cost(), which sends dataset dimensions and
consumes no quota. --live (or TABPFN_LIVE=1) adds one fit on 200 rows and two
predict requests on 100 rows (~20k tokens: 10k minimum per request) to record
the server-side model version, the shape of output_type="full", and whether a
dense quantile grid agrees with quantiles reconstructed from the full output.

Run from the repo root: `make check-api` (add LIVE=1 for the live call), or
`uv run python -m experiments.00_check_api [--live | --from-saved]`.
Writes results/00_api_check.json and, with --live, results/00_api_check_outputs.npz.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime

import numpy as np
import tabpfn_client
from tabpfn_client import TabPFNRegressor, estimate_cost, get_api_usage
from tabpfn_client.estimator import _limit_for_model_path

from src import config, data

OUT = config.RESULTS_DIR / "00_api_check.json"
RAW_OUT = config.RESULTS_DIR / "00_api_check_outputs.npz"

QUANTILE_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
DECILES = np.round(np.arange(0.1, 1.0, 0.1), 1)  # the client's default quantile levels

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


def bar_probs(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def bar_mean(logits, borders):
    return bar_probs(logits) @ ((borders[:-1] + borders[1:]) / 2)


def bar_quantiles(logits, borders, levels):
    """Invert the piecewise-uniform CDF; returns (n_rows, n_levels).

    The outer buckets are treated as uniform too, so extreme quantiles may
    differ from the server's if it models the tails differently.
    """
    p = bar_probs(logits)
    cdf = np.concatenate([np.zeros((len(p), 1)), np.cumsum(p, axis=1)], axis=1)
    widths = np.diff(borders)
    out = np.empty((len(p), len(levels)))
    for i in range(len(p)):
        k = np.clip(np.searchsorted(cdf[i], levels, side="right") - 1, 0, len(widths) - 1)
        frac = np.clip((levels - cdf[i, k]) / np.maximum(p[i, k], 1e-12), 0, 1)
        out[i] = borders[k] + frac * widths[k]
    return out


def as_rows_by_levels(q, n_rows):
    q = np.asarray(q)
    return q if q.shape[0] == n_rows else q.T


def abs_diff(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return {"max_K": float(d.max()), "median_K": float(np.median(d))}


def live_calls(engineered, target):
    """One fit and two predict requests on the same 100 rows: output_type="full"
    and a dense quantile grid. "full" is capped at 400 rows per request, so later
    phases want the quantile route if it agrees with the full distribution.

    Raw outputs are saved to RAW_OUT before anything else touches them, so an
    analysis bug can be fixed and rerun with --from-saved at no API cost.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(engineered), size=300, replace=False)
    train_idx, test_idx = idx[:200], idx[200:]
    X_test = engineered.iloc[test_idx]

    reg = TabPFNRegressor(model_path=config.TABPFN_MODEL_PATH, random_state=0)
    reg.fit(engineered.iloc[train_idx], target.iloc[train_idx])
    full = reg.predict(X_test, output_type="full")
    # The docs FAQ names this `last_meta`; tabpfn-client 0.6.1 only has `_last_meta`.
    meta_full, timings_full = dict(reg._last_meta), reg.get_timings()
    dense = reg.predict(X_test, output_type="quantiles", quantiles=QUANTILE_GRID.tolist())
    meta_dense = dict(reg._last_meta)

    arrays = {f"full_{k}": np.asarray(v) for k, v in full.items()}
    np.savez_compressed(
        RAW_OUT,
        test_idx=test_idx,
        y_test=target.iloc[test_idx].to_numpy(),
        dense_levels=QUANTILE_GRID,
        dense_quantiles=np.asarray(dense),
        **{k: v for k, v in arrays.items() if v.dtype.kind in "fiub"},
    )
    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "meta_full": meta_full,
        "meta_quantiles": meta_dense,
        "timings_full": timings_full,
        "full_output": {k: describe(v) for k, v in full.items()},
        "quantiles_output": describe(dense),
    }


def analyze(saved):
    """Compare quantiles and mean reconstructed from logits/borders with the server's."""
    logits = saved["full_logits"].astype(np.float64)
    borders = saved["full_borders"].astype(np.float64)
    n = len(saved["test_idx"])
    server_q9 = as_rows_by_levels(saved["full_quantiles"], n)
    dense_q = as_rows_by_levels(saved["dense_quantiles"], n)
    levels = saved["dense_levels"]
    interior = (levels >= 0.05) & (levels <= 0.95)
    at_deciles = np.isin(levels, DECILES)
    ours_dense = bar_quantiles(logits, borders, levels)
    return {
        "n_buckets": int(logits.shape[1]),
        "borders_shape": list(borders.shape),
        "borders_monotonic": bool(np.all(np.diff(borders) > 0)),
        "borders_range_K": [float(borders.min()), float(borders.max())],
        "mean_all_finite": bool(np.isfinite(saved["full_mean"]).all()),
        "reconstructed_vs_server": {
            "mean": abs_diff(bar_mean(logits, borders), saved["full_mean"]),
            "deciles_from_full": abs_diff(bar_quantiles(logits, borders, DECILES), server_q9),
            "dense_grid_interior_5_95": abs_diff(ours_dense[:, interior], dense_q[:, interior]),
            "dense_grid_all_levels": abs_diff(ours_dense, dense_q),
        },
        "full_vs_quantiles_request_at_deciles": abs_diff(server_q9, dense_q[:, at_deciles]),
    }


def write_report(report):
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"Wrote {OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="also run the ~20k-token live check")
    parser.add_argument(
        "--from-saved", action="store_true", help="redo the analysis from RAW_OUT, no API calls"
    )
    args = parser.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if args.from_saved:
        report = json.loads(OUT.read_text())
        report["live_smoke_check"].pop("analysis_error", None)
        report["live_smoke_check"]["analysis"] = analyze(np.load(RAW_OUT))
        print(json.dumps(report["live_smoke_check"]["analysis"], indent=2))
        write_report(report)
        return

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
    # init() authenticates and fetches server settings. _limit_for_model_path is
    # private, but it is where the client reads the server's size limits,
    # including the per-request row cap for output_type="full".
    tabpfn_client.init()
    report["server_limits"] = _limit_for_model_path(config.TABPFN_MODEL_PATH).model_dump()

    for name, X_train, X_test in cost_scenarios(engineered, composition):
        quote = estimate_cost(X_train, X_test, model_version=config.TABPFN_MODEL_VERSION)
        report["cost_estimates"][name] = quote.model_dump(mode="json")
        print(f"{name:32s} {X_train.shape} -> {len(X_test)} rows: {quote.estimated_cost:,} tokens")

    if not live:
        print("Skipped live fit/predict (pass --live or set TABPFN_LIVE=1).")
        write_report(report)
        return

    report["usage_before"] = get_api_usage()
    report["live_smoke_check"] = live_calls(engineered, target)
    report["usage_after"] = get_api_usage()
    try:
        report["live_smoke_check"]["analysis"] = analyze(np.load(RAW_OUT))
    except Exception as exc:
        report["live_smoke_check"]["analysis_error"] = repr(exc)
        write_report(report)
        raise
    print(json.dumps(report["live_smoke_check"], indent=2, default=str))
    write_report(report)


if __name__ == "__main__":
    main()
