"""Phase 4 discovery loop, end to end: pilot -> gates -> winner -> main and hard runs.

  all        pilot runs (EI and q90, seeds 100-102, both scenarios) and pilot baselines;
             winner by the pre-registered rule (discovery.pilot_decision); gates; then the
             main runs (winner(s) + greedy TabPFN, seeds 0-9, both scenarios) and baselines
  pilot | baselines | gates   one stage only

Gates between pilot and main (any failure stops the run before a main request is sent):
  G1 every pilot run completed all rounds with no error and no missing round
  G2 simulated random search is within 4 standard errors of its exact (hypergeometric)
     mean and probability of at least one hit, in both scenarios
  G3 every selected batch came only from unlabeled pool materials (and the start sets
     hold no target; Tc and target labels match the data)
  G4 the budget guard authorizes the main runs
None of them looks at which method finds more targets.

Committed: results/04_discovery/runs/<phase>/<scenario>/<acquisition>/seed_SSS.json (per-round
summaries), pilot_decision.json, gates.json. Local only (results/cache/discovery/): per-material
scores for every round and raw quantile grids for the first and last round. Every stage
resumes: finished rounds are kept and unfinished runs continue.

TabPFN requests need --live and TABPFN_TOKEN. --dry-run swaps in discovery.fake_predict and
sends every output, the ledger and the usage figures to results/cache/dryrun/ (no API).

Overnight: `caffeinate -is uv run python -m experiments.04_discovery all --live --workers 4`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src import benchmark, budget, config, discovery, models

RES = config.RESULTS_DIR
LINES = {"pilot": "p4_pilot", ("main", "main"): "p4_main", ("main", "hard"): "p4_hard"}
PILOT_ACQ = ("ei", "q90")
RANDOM_REPLICATES = 1000
RETRY_WAIT_S = 60
FAKE_USAGE = {
    "monthly_token_limit": 20_000_000,
    "monthly_tokens_used": 0,
    "daily_token_limit": 15_000_000,
    "daily_tokens_used": 0,
}


class GateFailed(RuntimeError):
    pass


@dataclass
class Ctx:
    live: bool
    dry: bool
    workers: int
    rounds: int = discovery.ROUNDS
    pilot_seeds: tuple = discovery.PILOT_SEEDS
    main_seeds: tuple = discovery.MAIN_SEEDS
    out: Path = field(default=RES / "04_discovery")
    local: Path = field(default=RES / "cache" / "discovery")

    def authorize(self, line, n, tokens):
        if self.dry:
            return budget.authorize(
                line, n, tokens, usage=FAKE_USAGE, ledger=self.local / "ledger.jsonl"
            )
        return budget.authorize(line, n, tokens)


def line_for(phase, scenario):
    return LINES["pilot"] if phase == "pilot" else LINES[("main", scenario)]


def run_path(ctx, phase, scenario, acq, seed) -> Path:
    return ctx.out / "runs" / phase / scenario / acq / f"seed_{seed:03d}.json"


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def save(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(tmp, path)


_SCENARIOS: dict = {}


def scenario(name) -> discovery.Scenario:
    if name not in _SCENARIOS:
        _SCENARIOS[name] = discovery.load_scenario(name)
    return _SCENARIOS[name]


def persister(ctx, phase, path, rounds):
    local = ctx.local / path.relative_to(ctx.out / "runs").with_suffix("")

    def on_round(record, rnd, scores, raw):
        save(path, record)
        local.mkdir(parents=True, exist_ok=True)
        scores.astype({c: "float32" for c in scores.columns if c != "pool_index"}).to_parquet(
            local / f"round_{rnd:02d}.parquet", compression="zstd", index=False
        )
        if raw is not None and rnd in (1, rounds):
            np.savez_compressed(
                local / f"raw_round_{rnd:02d}.npz",
                mean=raw["mean"],
                quantiles=np.asarray(raw["quantiles"], np.float32),
                levels=discovery.TAIL_LEVELS,
                pool_index=scores["pool_index"].to_numpy(),
            )

    return on_round


def tabpfn_job(ctx, phase, scenario_name, acq, seed, run_budget, retries):
    """One TabPFN discovery run in a worker process; charges before every request."""
    scn = scenario(scenario_name)
    path = run_path(ctx, phase, scenario_name, acq, seed)
    record = load(path)
    if record is not None:
        if record.get("status") == "complete":
            return path, "cached", None
        record.pop("error", None)
        record["status"] = "running"

    def predict(X_train, y_train, X_pool, s):
        for attempt in range(retries + 1):
            run_budget.charge(1, run=f"{phase}/{scenario_name}/{acq}/{seed}", attempt=attempt)
            t = time.perf_counter()
            try:
                if ctx.dry:
                    out = discovery.fake_predict(X_train, y_train, X_pool, s)
                else:
                    out = models.tabpfn_predict(
                        X_train.reset_index(drop=True),
                        y_train,
                        X_pool.reset_index(drop=True),
                        s,
                        levels=discovery.TAIL_LEVELS,
                    )
                out["seconds"] = round(time.perf_counter() - t, 1)
                return out
            except budget.BudgetExceeded:
                raise
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(RETRY_WAIT_S)

    try:
        record = discovery.run_loop(
            scn,
            acq,
            seed,
            predict,
            record,
            persister(ctx, phase, path, ctx.rounds),
            rounds=ctx.rounds,
        )
        record["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        save(path, record)
        return path, "complete", None
    except Exception as exc:
        rec = load(path) or {
            "scenario": scenario_name,
            "acquisition": acq,
            "seed": seed,
            "rounds": [],
        }
        rec["status"], rec["error"] = "error", repr(exc)[:500]
        save(path, rec)
        return path, "error", repr(exc)[:300]


def run_tabpfn(ctx, phase, acqs, seeds) -> list:
    jobs = [(sc, a, s) for sc in discovery.SCENARIOS for a in acqs for s in seeds]
    todo = []
    for sc, a, s in jobs:
        rec = load(run_path(ctx, phase, sc, a, s))
        if rec is None or rec.get("status") != "complete":
            todo.append((sc, a, s, ctx.rounds - len((rec or {}).get("rounds", []))))
    print(f"[{phase}] {len(jobs) - len(todo)} of {len(jobs)} TabPFN runs complete", flush=True)
    if not todo:
        return []
    if not (ctx.live or ctx.dry):
        raise SystemExit("TabPFN runs need --live (or --dry-run).")
    if ctx.live:
        config.require_tabpfn_token()
    slices = {}
    for line in sorted({line_for(phase, sc) for sc, *_ in todo}):
        group = [t for t in todo if line_for(phase, t[0]) == line]
        base = sum(t[3] for t in group)
        if ctx.dry:
            tokens = budget.MIN_TOKENS_PER_REQUEST
        else:
            scn = scenario(group[0][0])
            tokens = budget.estimate_tokens(scn.X.iloc[:250], scn.X)
        plan = budget.PLAN_BY_NAME[line]
        spent = sum(
            e["requests"]
            for e in budget.read_ledger(ctx.local / "ledger.jsonl" if ctx.dry else budget.LEDGER)
            if e["experiment"] == line
        )
        retries = max(0, min(1, (plan.cap - spent - base) // len(group)))
        auth = ctx.authorize(line, base + retries * len(group), tokens)
        print(
            f"[{phase}] {line}: authorized {base + retries * len(group)} requests "
            f"({retries} retry per run)",
            flush=True,
        )
        for sc, a, s, remaining in group:
            slices[sc, a, s] = (
                budget.RunBudget(line, remaining + retries, tokens, auth.ledger, auth.record),
                retries,
            )
    results, start = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=ctx.workers) as pool:
        futures = {
            pool.submit(tabpfn_job, ctx, phase, sc, a, s, *slices[sc, a, s]): (sc, a, s)
            for sc, a, s, _ in todo
        }
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                path, status, err = fut.result()
                results.append((path, status, err))
                print(
                    f"[{phase}] {'/'.join(map(str, futures[fut]))}: {status}"
                    f"{' ' + err if err else ''} ({time.perf_counter() - start:.0f}s)",
                    flush=True,
                )
    return results


def run_baselines(ctx, phase, seeds) -> None:
    for sc in discovery.SCENARIOS:
        scn = scenario(sc)
        for acq in discovery.FREE_ACQUISITIONS:
            for s in seeds:
                path = run_path(ctx, phase, sc, acq, s)
                rec = load(path)
                if rec is not None and rec.get("status") == "complete":
                    continue

                def xgb_predict(X_train, y_train, X_pool, seed):
                    m = models.hamidieh_xgb(seed).fit(X_train.to_numpy(np.float32), y_train)
                    return {
                        "mean": m.predict(X_pool.to_numpy(np.float32)),
                        "quantiles": None,
                        "meta": {"model_path": "xgboost published settings"},
                    }

                rec = discovery.run_loop(
                    scn,
                    acq,
                    s,
                    xgb_predict,
                    rec,
                    lambda r, *_, p=path: save(p, r),
                    rounds=ctx.rounds,
                )
                rec["machine"] = benchmark.machine_info()
                save(path, rec)
        print(f"[{phase}] baselines done for {sc}", flush=True)


def found_last(rec) -> int:
    return rec["rounds"][-1]["found_cumulative"]


def decide(ctx) -> dict:
    found = {a: [] for a in PILOT_ACQ}
    for sc in discovery.SCENARIOS:
        for a in PILOT_ACQ:
            for s in ctx.pilot_seeds:
                found[a].append(found_last(load(run_path(ctx, "pilot", sc, a, s))))
    decision = discovery.pilot_decision(found) | {
        "per_run_found": found,
        "seeds": list(ctx.pilot_seeds),
        "rounds": ctx.rounds,
        "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    save(ctx.out / "pilot_decision.json", decision)
    print(f"[decision] totals {decision['totals']} -> {decision['chosen']}", flush=True)
    return decision


def gates(ctx, decision) -> dict:
    report, failures = {}, []
    # G1 pilot runs complete
    g1 = []
    for sc in discovery.SCENARIOS:
        for a in PILOT_ACQ:
            for s in ctx.pilot_seeds:
                rec = load(run_path(ctx, "pilot", sc, a, s))
                if rec is None:
                    g1.append(f"pilot/{sc}/{a}/seed {s}: missing")
                elif rec.get("status") != "complete" or "error" in rec:
                    g1.append(
                        f"pilot/{sc}/{a}/seed {s}: {rec.get('status')} {rec.get('error', '')}"
                    )
                elif len(rec["rounds"]) != ctx.rounds:
                    g1.append(f"pilot/{sc}/{a}/seed {s}: {len(rec['rounds'])} rounds")
    report["G1_pilot_complete"] = {"pass": not g1, "problems": g1}
    # G3 integrity of every pilot record, TabPFN and baselines
    g3 = []
    for sc in discovery.SCENARIOS:
        scn = scenario(sc)
        for a in (*PILOT_ACQ, *discovery.FREE_ACQUISITIONS):
            for s in ctx.pilot_seeds:
                rec = load(run_path(ctx, "pilot", sc, a, s))
                if rec is not None:
                    g3 += discovery.check_run(rec, scn, rounds=ctx.rounds)
    report["G3_batches_from_unlabeled"] = {"pass": not g3, "problems": g3[:50]}
    # G2 random search simulator vs exact expectation
    g2 = {}
    for sc in discovery.SCENARIOS:
        scn = scenario(sc)
        sims = discovery.simulate_random(scn, RANDOM_REPLICATES, rounds=ctx.rounds)
        exact = discovery.random_expectation(scn, ctx.rounds * discovery.BATCH)
        se_mean = exact["sd"] / np.sqrt(len(sims))
        p = exact["p_at_least_one"]
        se_p = np.sqrt(p * (1 - p) / len(sims))
        ok = (
            abs(sims.mean() - exact["mean"]) <= 4 * se_mean
            and abs((sims > 0).mean() - p) <= 4 * se_p
        )
        g2[sc] = {
            "pass": bool(ok),
            "replicates": len(sims),
            "simulated_mean": float(sims.mean()),
            "exact_mean": exact["mean"],
            "simulated_p_hit": float((sims > 0).mean()),
            "exact_p_hit": p,
            "tolerance": "4 standard errors",
        }
    report["G2_random_vs_exact"] = {"pass": all(v["pass"] for v in g2.values()), **g2}
    # G4 budget guard authorizes the main runs
    acqs = [*decision["chosen"], "greedy_tabpfn"]
    n = len(acqs) * len(ctx.main_seeds) * ctx.rounds
    g4 = {"acquisitions": acqs, "requests_per_scenario": n}
    try:
        for sc in discovery.SCENARIOS:
            ctx.authorize(line_for("main", sc), n, budget.MIN_TOKENS_PER_REQUEST)
        g4["pass"] = True
    except budget.BudgetExceeded as exc:
        g4 |= {"pass": False, "problem": str(exc)}
    report["G4_budget_authorizes_main"] = g4
    for name, g in report.items():
        if not g["pass"]:
            failures.append(name)
    report["all_pass"] = not failures
    report["checked_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    save(ctx.out / "gates.json", report)
    print(f"[gates] {'all pass' if not failures else 'FAILED: ' + ', '.join(failures)}", flush=True)
    if failures:
        raise GateFailed(f"gates failed: {failures}; see {ctx.out / 'gates.json'}")
    return report


def final_check(ctx, decision) -> None:
    problems = []
    acqs = [*decision["chosen"], "greedy_tabpfn", *discovery.FREE_ACQUISITIONS]
    for sc in discovery.SCENARIOS:
        scn = scenario(sc)
        for a in acqs:
            for s in ctx.main_seeds:
                rec = load(run_path(ctx, "main", sc, a, s))
                if rec is None:
                    problems.append(f"main/{sc}/{a}/seed {s}: missing")
                else:
                    problems += discovery.check_run(rec, scn, rounds=ctx.rounds)
    report = load(ctx.out / "gates.json")
    report["final_main_integrity"] = {"pass": not problems, "problems": problems[:50]}
    save(ctx.out / "gates.json", report)
    print(
        f"[final] main runs: {'all valid' if not problems else f'{len(problems)} problems'}",
        flush=True,
    )
    if problems:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "pilot", "baselines", "gates"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=None, help="dry runs only")
    parser.add_argument("--pilot-seeds", type=int, nargs="+", default=None, help="dry runs only")
    parser.add_argument("--main-seeds", type=int, nargs="+", default=None, help="dry runs only")
    args = parser.parse_args()
    if args.live and args.dry_run:
        raise SystemExit("--live and --dry-run are exclusive")
    ctx = Ctx(
        live=config.live_requested(args.live) and not args.dry_run,
        dry=args.dry_run,
        workers=args.workers,
    )
    if args.dry_run:
        ctx.out = RES / "cache" / "dryrun" / "04_discovery"
        ctx.local = RES / "cache" / "dryrun" / "local"
        ctx.rounds = args.rounds or ctx.rounds
        ctx.pilot_seeds = tuple(args.pilot_seeds or ctx.pilot_seeds)
        ctx.main_seeds = tuple(args.main_seeds or ctx.main_seeds)
    elif args.rounds or args.pilot_seeds or args.main_seeds:
        raise SystemExit("--rounds/--seeds overrides are for --dry-run only")

    start = time.perf_counter()
    if args.stage in ("all", "pilot"):
        run_tabpfn(ctx, "pilot", PILOT_ACQ, ctx.pilot_seeds)
        run_baselines(ctx, "pilot", ctx.pilot_seeds)
    if args.stage == "baselines":
        run_baselines(ctx, "pilot", ctx.pilot_seeds)
        run_baselines(ctx, "main", ctx.main_seeds)
    if args.stage in ("all", "gates"):
        decision = decide(ctx)
        gates(ctx, decision)
    if args.stage == "all":
        run_tabpfn(ctx, "main", [*decision["chosen"], "greedy_tabpfn"], ctx.main_seeds)
        run_baselines(ctx, "main", ctx.main_seeds)
        final_check(ctx, decision)
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
