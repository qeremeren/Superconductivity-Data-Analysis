"""Phase 2 TabPFN-3.5 runs: one billed predict request per (split, feature set).

Without --live this only reads the committed per-row summaries in
results/02_benchmark/tabpfn/ and fails if any are missing, so `make reproduce`
never calls the API. With --live (and TABPFN_TOKEN) it requests only what is
missing: each split kind is authorized against its budget line first, every
request is charged in this process before a worker sends it, raw outputs are
saved before they are summarized, and a failed job is reported without stopping
the others. Rerunning resumes where it stopped.

Run from the repo root:
  uv run python -m experiments.02_tabpfn [--live] [--kinds ...] [--splits 0 1] [--workers 4]
"""

from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from src import benchmark, budget, config, data, models

BUDGET_LINES = {
    "random": "p2_random_protocol",
    "grouped": "p2_grouped",
    "grouped_no_oxygen": "p2_grouped_no_oxygen",
    "leave_family_out": "p2_leave_family_out",
}
CACHE = models.TabPFNCache(benchmark.OUT / "tabpfn")


def build_jobs(inputs, kinds, features, only_splits=None) -> list[models.TabPFNJob]:
    jobs = []
    for kind, split, seed, is_test in benchmark.iter_splits(kinds):
        if only_splits is not None and split not in only_splits:
            continue
        for fs in features:
            X = inputs.features[fs]
            jobs.append(
                models.TabPFNJob(
                    name=f"{kind}/{fs}/split_{split:02d}",
                    seed=seed,
                    X_train=X[~is_test].reset_index(drop=True),
                    y_train=inputs.y[~is_test],
                    X_test=X[is_test].reset_index(drop=True),
                    y_test=inputs.y[is_test],
                    test_rows=np.flatnonzero(is_test),
                )
            )
    return jobs


def run(
    jobs,
    live: bool,
    workers: int,
    predict=models.tabpfn_predict,
    authorize=budget.authorize,
    estimate=budget.estimate_tokens,
    cache=CACHE,
) -> dict:
    missing = [job for job in jobs if cache.load(job) is None]
    print(f"{len(jobs) - len(missing)} of {len(jobs)} jobs cached, {len(missing)} missing")
    if not missing:
        return {"requested": 0, "failed": [], "not_sent": []}
    if not live:
        raise models.MissingCache(
            f"{len(missing)} TabPFN results are not cached (first: {missing[0].name}). "
            "Reproducing never calls the API; rerun with --live and TABPFN_TOKEN to request them."
        )
    config.require_tabpfn_token()

    budgets = {}
    for kind in sorted({job.name.split("/")[0] for job in missing}):
        todo = [job for job in missing if job.name.startswith(kind + "/")]
        biggest = max(todo, key=lambda j: (len(j.X_train), len(j.X_test), j.X_train.shape[1]))
        tokens = estimate(biggest.X_train, biggest.X_test)
        budgets[kind] = authorize(BUDGET_LINES[kind], len(todo), tokens)
        print(f"{kind}: authorized {len(todo)} requests at {tokens:,} tokens each")

    failed, start = [], time.perf_counter()

    def finish(job, result):
        cache.save_raw(job, result)
        cache.load(job)

    queue, sent, done = list(missing), 0, 0

    def charge(job):
        budgets[job.name.split("/")[0]].charge(1, job=job.name, output_type="main")

    if workers <= 1:
        for job in queue:
            charge(job)
            sent += 1
            try:
                finish(job, predict(job.X_train, job.y_train, job.X_test, job.seed))
            except Exception:
                failed.append(job.name)
                traceback.print_exc()
        queue = []
    else:
        # At most `workers` jobs are in flight, each charged when it is dispatched, so a
        # broken pool leaves at most `workers` charged-but-unsent jobs in the ledger.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures, broken = {}, False
            while queue or futures:
                while queue and len(futures) < workers and not broken:
                    job = queue.pop(0)
                    charge(job)
                    sent += 1
                    futures[
                        pool.submit(predict, job.X_train, job.y_train, job.X_test, job.seed)
                    ] = job
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in finished:
                    job, done = futures.pop(fut), done + 1
                    try:
                        finish(job, fut.result())
                        elapsed = time.perf_counter() - start
                        print(f"[{done}/{len(missing)}] {job.name} ({elapsed:.0f}s)", flush=True)
                    except BrokenProcessPool:
                        failed.append(job.name)
                        broken = True
                    except Exception:
                        failed.append(job.name)
                        print(f"[{done}/{len(missing)}] {job.name} FAILED", flush=True)
                        traceback.print_exc()
                if broken and not futures:
                    print("Worker pool broke; stopping. Unsent jobs were not charged.", flush=True)
                    break
    print(f"Sent {sent}, failed {len(failed)}, not sent {len(queue)}: {failed}")
    return {"requested": sent, "failed": failed, "not_sent": [job.name for job in queue]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--kinds", nargs="+", default=list(benchmark.SPLIT_KINDS))
    parser.add_argument("--features", nargs="+", default=list(data.FEATURE_SETS))
    parser.add_argument("--splits", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    inputs = benchmark.load_inputs()
    jobs = build_jobs(inputs, args.kinds, args.features, args.splits)
    out = run(jobs, config.live_requested(args.live), args.workers)
    if out["failed"] or out.get("not_sent"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
