"""Phase 2 optional add-on: TabPFN-3.5-Thinking with group_col on the grouped split.

Thinking spends extra compute at fit time on an internal validation, and group_col
tells it which rows are the same material, so that validation never splits one.
The group column is an integer material ID (the factorized composition key), not
the composition string, so it adds no composition information if the server also
uses it as a feature. Composition features (standard TabPFN's best set), medium
effort, RMSE as the thinking metric.

--probe runs grouped split 0 only. Without it: grouped splits 0-9. Each split is
two billed requests (a thinking fit and a predict), each charged before it is sent.
Same caching rules as 02_tabpfn: committed per-row summaries under
results/02_benchmark/thinking/, raw grids local, and no API call without --live.

Run from the repo root: `uv run python -m experiments.02_thinking --live [--probe]`.
"""

from __future__ import annotations

import argparse
import importlib
import time
import traceback

import numpy as np
import pandas as pd

from src import benchmark, budget, config, metrics, models

GROUP_COL = "material_id"
FEATURES = "composition"
THINKING = {"thinking_effort": "medium", "thinking_metric": "rmse", "thinking_timeout_s": 900}
SPLITS = tuple(range(10))
CACHE = models.TabPFNCache(benchmark.OUT / "thinking")


def build_jobs(inputs, splits) -> list[models.TabPFNJob]:
    material_id = pd.factorize(inputs.composition_key, sort=True)[0]
    jobs = []
    for _kind, split, seed, is_test in benchmark.iter_splits(["grouped"]):
        if split not in splits:
            continue
        X = inputs.features[FEATURES].assign(**{GROUP_COL: material_id})
        jobs.append(
            models.TabPFNJob(
                name=f"grouped/{FEATURES}/split_{split:02d}",
                seed=seed,
                X_train=X[~is_test].reset_index(drop=True),
                y_train=inputs.y[~is_test],
                X_test=X[is_test].reset_index(drop=True),
                y_test=inputs.y[is_test],
                test_rows=np.flatnonzero(is_test),
            )
        )
    return jobs


def thinking_predict(job: models.TabPFNJob, run_budget, regressor=None) -> dict:
    """Thinking fit and one predict; each billed request is charged before it is sent."""
    if regressor is None:
        regressor = importlib.import_module("tabpfn_client").TabPFNRegressor
    reg = regressor(
        model_path=config.TABPFN_MODEL_PATH,
        random_state=int(job.seed),
        group_col=GROUP_COL,
        **THINKING,
    )
    run_budget.charge(1, job=job.name, op="thinking_fit")
    t = time.perf_counter()
    reg.fit(job.X_train, job.y_train)
    fit_seconds = time.perf_counter() - t
    run_budget.charge(1, job=job.name, op="thinking_predict", output_type="main")
    t = time.perf_counter()
    out = reg.predict(
        job.X_test, output_type="main", quantiles=[float(x) for x in metrics.QUANTILE_LEVELS]
    )
    predict_seconds = time.perf_counter() - t
    q = np.asarray(out["quantiles"], dtype=np.float64)
    if q.shape == (len(metrics.QUANTILE_LEVELS), len(job.X_test)):
        q = q.T
    return {
        "mean": np.asarray(out["mean"], dtype=np.float64),
        "quantiles": q,
        "meta": {
            **dict(reg._last_meta),
            "thinking": THINKING,
            "group_col": GROUP_COL,
            "fit_wall_seconds": round(fit_seconds, 1),
            "predict_wall_seconds": round(predict_seconds, 1),
        },
        "timings": reg.get_timings(),
    }


def run(
    jobs,
    live: bool,
    experiment: str,
    regressor=None,
    authorize=budget.authorize,
    estimate=None,
    cache=CACHE,
) -> dict:
    missing = [job for job in jobs if cache.load(job) is None]
    print(
        f"{len(jobs) - len(missing)} of {len(jobs)} jobs cached, {len(missing)} missing", flush=True
    )
    if not missing:
        return {"sent_jobs": 0, "failed": []}
    if not live:
        raise models.MissingCache(
            f"{len(missing)} Thinking results are not cached; rerun with --live to request them."
        )
    config.require_tabpfn_token()
    tokens = (estimate or thinking_tokens)(missing[0])
    run_budget = authorize(experiment, 2 * len(missing), tokens)
    print(f"authorized {2 * len(missing)} requests at {tokens:,} tokens each", flush=True)
    failed = []
    for job in missing:
        try:
            result = thinking_predict(job, run_budget, regressor)
            cache.save_raw(job, result)
            cache.load(job)
            print(
                f"{job.name}: fit {result['meta']['fit_wall_seconds']}s, "
                f"predict {result['meta']['predict_wall_seconds']}s",
                flush=True,
            )
        except budget.BudgetExceeded:
            raise
        except Exception:
            failed.append(job.name)
            print(f"{job.name} FAILED", flush=True)
            traceback.print_exc()
    return {"sent_jobs": len(missing), "failed": failed}


def thinking_tokens(job) -> int:
    """The larger of the two quoted charges (the predict), so the cap is conservative."""
    from tabpfn_client import estimate_cost

    fit = estimate_cost(
        job.X_train,
        operation="thinking_fit",
        model_version="v3.5",
        thinking_effort=THINKING["thinking_effort"],
    )
    pred = estimate_cost(
        job.X_train, job.X_test, operation="thinking_predict", model_version="v3.5", n_estimators=8
    )
    return max(budget.MIN_TOKENS_PER_REQUEST, int(fit.estimated_cost), int(pred.estimated_cost))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--probe", action="store_true", help="grouped split 0 only")
    args = parser.parse_args()
    jobs = build_jobs(benchmark.load_inputs(), (0,) if args.probe else SPLITS)
    experiment = "p2_thinking_probe" if args.probe else "p2_thinking"
    out = run(jobs, config.live_requested(args.live), experiment)
    print(out, flush=True)
    if out["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
