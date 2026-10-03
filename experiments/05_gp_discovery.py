"""Phase 5 discovery baseline: Gaussian-process EI (no API).

Same scenarios, seeds 0-9, start sets, 20 rounds of 10 and y* as the Phase 4 main runs; the
GP configuration (models.gp_predict) was fixed in the Phase 5 plan before any GP run. Not part
of the pre-registered Phase 4 design. Records go next to the Phase 4 runs, at
results/04_discovery/runs/main/<scenario>/gp_ei/seed_SSS.json, so experiments.04_evaluate
checks and evaluates them with the rest. Resumable: complete records are skipped.

Run from the repo root: `uv run python -m experiments.05_gp_discovery`.
"""

from __future__ import annotations

import json
import os

from src import benchmark, config, discovery, models

RUNS = config.RESULTS_DIR / "04_discovery" / "runs" / "main"
DESIGN = (
    "Phase 5 baseline: GP-EI, configuration fixed in the Phase 5 plan before any GP run; "
    "not part of the pre-registered Phase 4 design"
)


def save(path, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(tmp, path)


def main():
    for sc in discovery.SCENARIOS:
        scn = discovery.load_scenario(sc)
        for seed in discovery.MAIN_SEEDS:
            path = RUNS / sc / "gp_ei" / f"seed_{seed:03d}.json"
            rec = json.loads(path.read_text()) if path.is_file() else None
            if rec is not None and rec.get("status") == "complete":
                continue
            rec = discovery.run_loop(
                scn, "gp_ei", seed, models.gp_predict, rec, lambda r, *_, p=path: save(p, r)
            )
            problems = discovery.check_run(rec, scn)
            if problems:
                raise SystemExit(f"{sc} seed {seed}: {problems[0]}")
            rec["design"] = DESIGN
            rec["machine"] = benchmark.machine_info()
            save(path, rec)
            print(f"{sc} seed {seed}: {rec['rounds'][-1]['found_cumulative']} targets", flush=True)


if __name__ == "__main__":
    main()
