"""Phase 4 evaluation from the committed run records only (no API calls, no fitting).

Reads results/04_discovery/runs/<phase>/<scenario>/<acquisition>/seed_SSS.json and writes to
results/04_discovery/:
  curves.csv       mean targets found (95% t-interval over seeds) after each round, with the
                   exact random-search expectation
  summary.csv      per phase, scenario and method: targets found at 50/100/200 experiments,
                   tries to the first hit (median, runs without a hit), enrichment over random
  paired.csv       per-seed differences in targets found after 50/100/200 experiments, EI vs
                   every other method (EI vs greedy TabPFN = same model with vs without
                   uncertainty)
  paired_seeds.csv targets found per seed and method after 200 experiments (main runs)
  stalls.csv       post hoc: runs that stalled (at most discovery.STALL_MAX targets after 200
                   experiments; random expects ~2) per method, and on which seeds
  paired_by_stall.csv  EI vs each greedy method, split by whether the greedy run stalled
  ablation_check.json  EI and greedy TabPFN start from the same training set and the same first
                   fit: identical round-1 fingerprints and predicted means on shared picks
  novelty.csv      how new the found targets were: L1 distance (element fractions) to the
                   nearest material labeled before it was picked, and whether it was only an
                   oxygen variant of a labeled material
  families.csv     families of the found targets
  reliability.csv  predicted P(top 1%) vs outcome over the unlabeled pool, all TabPFN rounds
  batch_calibration.csv  expected (sum of P) vs realized targets and Tc > 77 K hits per batch
  timing.json      request seconds
`--check` re-runs the integrity checks on every record first (used by make reproduce).

Run from the repo root: `uv run python -m experiments.04_evaluate [--check]`.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config, data, discovery

OUT = config.RESULTS_DIR / "04_discovery"
CHECKPOINTS = (50, 100, 200)
GREEDY = ("greedy_tabpfn", "greedy_xgb")


def load_runs() -> list[dict]:
    runs = []
    for path in sorted((OUT / "runs").glob("*/*/*/seed_*.json")):
        rec = json.loads(path.read_text())
        rec["phase"] = path.parts[-4]
        runs.append(rec)
    return runs


def first_hit(rec) -> int | None:
    for r in rec["rounds"]:
        for j, s in enumerate(r["selected"]):
            if s["is_target"]:
                return (r["round"] - 1) * rec["batch"] + j + 1
    return None


def ci95(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def curves(runs, scns) -> pd.DataFrame:
    rows = []
    groups = {}
    for rec in runs:
        groups.setdefault((rec["phase"], rec["scenario"], rec["acquisition"]), []).append(rec)
    for (phase, sc, acq), recs in groups.items():
        for i in range(len(recs[0]["rounds"])):
            exp_n = (i + 1) * recs[0]["batch"]
            found = [r["rounds"][i]["found_cumulative"] for r in recs]
            m, lo, hi = ci95(found)
            rows.append(
                {
                    "phase": phase,
                    "scenario": sc,
                    "acquisition": acq,
                    "experiments": exp_n,
                    "seeds": len(recs),
                    "found_mean": m,
                    "found_lo": lo,
                    "found_hi": hi,
                    "random_expected": discovery.random_expectation(scns[sc], exp_n)["mean"],
                }
            )
    return pd.DataFrame(rows)


def summary(runs, scns) -> pd.DataFrame:
    rows = []
    groups = {}
    for rec in runs:
        groups.setdefault((rec["phase"], rec["scenario"], rec["acquisition"]), []).append(rec)
    for (phase, sc, acq), recs in groups.items():
        hits = [first_hit(r) for r in recs]
        hit_tries = [h for h in hits if h is not None]
        row = {
            "phase": phase,
            "scenario": sc,
            "acquisition": acq,
            "seeds": len(recs),
            "targets_in_pool": recs[0]["targets_in_pool"],
            "first_hit_median": float(np.median(hit_tries)) if hit_tries else np.nan,
            "first_hit_min": min(hit_tries) if hit_tries else np.nan,
            "first_hit_max": max(hit_tries) if hit_tries else np.nan,
            "runs_without_hit": sum(h is None for h in hits),
            "random_expected_first_hit": (recs[0]["pool_size"] - recs[0]["n_initial"] + 1)
            / (recs[0]["targets_in_pool"] + 1),
        }
        for n in CHECKPOINTS:
            i = n // recs[0]["batch"] - 1
            found = [r["rounds"][i]["found_cumulative"] for r in recs]
            m, lo, hi = ci95(found)
            exp = discovery.random_expectation(scns[sc], n)["mean"]
            row |= {
                f"found_{n}": m,
                f"found_{n}_lo": lo,
                f"found_{n}_hi": hi,
                f"random_{n}": exp,
                f"enrichment_{n}": m / exp,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def found_at(rec, n) -> int:
    return rec["rounds"][n // rec["batch"] - 1]["found_cumulative"]


def paired(runs) -> pd.DataFrame:
    by = {(r["phase"], r["scenario"], r["acquisition"], r["seed"]): r for r in runs}
    rows = []
    for sc in discovery.SCENARIOS:
        seeds = sorted({k[3] for k in by if k[:3] == ("main", sc, "ei")})
        for other in ("greedy_tabpfn", "greedy_xgb", "random"):
            for n in CHECKPOINTS:
                d = np.array(
                    [
                        found_at(by["main", sc, "ei", s], n) - found_at(by["main", sc, other, s], n)
                        for s in seeds
                    ]
                )
                m, lo, hi = ci95(d)
                rows.append(
                    {
                        "phase": "main",
                        "scenario": sc,
                        "a": "ei",
                        "b": other,
                        "experiments": n,
                        "seeds": len(seeds),
                        "mean_diff": m,
                        "diff_lo": lo,
                        "diff_hi": hi,
                        "a_better": int((d > 0).sum()),
                        "b_better": int((d < 0).sum()),
                        "ties": int((d == 0).sum()),
                    }
                )
    return pd.DataFrame(rows)


def paired_seeds(runs) -> pd.DataFrame:
    rows = [
        {
            "scenario": r["scenario"],
            "seed": r["seed"],
            "acquisition": r["acquisition"],
            "found_200": found_at(r, 200),
        }
        for r in runs
        if r["phase"] == "main"
    ]
    return (
        pd.DataFrame(rows)
        .pivot_table(index=["scenario", "seed"], columns="acquisition", values="found_200")
        .reset_index()
    )


def stalls(seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, prow = [], []
    for sc, d in seeds.groupby("scenario"):
        for acq in ("ei", *GREEDY):
            st = d[acq] <= discovery.STALL_MAX
            rows.append(
                {
                    "scenario": sc,
                    "acquisition": acq,
                    "seeds": len(d),
                    "stalled": int(st.sum()),
                    "max_found_stalled": d[acq][st].max() if st.any() else np.nan,
                    "min_found_not_stalled": d[acq][~st].min() if (~st).any() else np.nan,
                    "stalled_seeds": " ".join(str(x) for x in d.seed[st]),
                }
            )
        for other in GREEDY:
            for stalled in (True, False):
                m = (d[other] <= discovery.STALL_MAX) == stalled
                diff = (d.ei - d[other])[m]
                prow.append(
                    {
                        "scenario": sc,
                        "b": other,
                        "b_stalled": stalled,
                        "seeds": int(m.sum()),
                        "ei_mean": d.ei[m].mean(),
                        "b_mean": d[other][m].mean(),
                        "mean_diff": diff.mean(),
                        "ei_better": int((diff > 0).sum()),
                        "b_better": int((diff < 0).sum()),
                        "ties": int((diff == 0).sum()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(prow)


def ablation_check(runs) -> dict:
    """EI vs greedy TabPFN: same training set and same fit in round 1, so only the rule differs."""
    by = {(r["phase"], r["scenario"], r["acquisition"], r["seed"]): r for r in runs}
    out = {}
    for sc in discovery.SCENARIOS:
        seeds = sorted({k[3] for k in by if k[:3] == ("main", sc, "ei")})
        same_set, shared, diffs = 0, 0, []
        for s in seeds:
            a = by["main", sc, "ei", s]["rounds"][0]
            b = by["main", sc, "greedy_tabpfn", s]["rounds"][0]
            same_set += a["labeled_fingerprint"] == b["labeled_fingerprint"]
            mean_a = {x["key"]: x["mean"] for x in a["selected"]}
            for x in b["selected"]:
                if x["key"] in mean_a:
                    shared += 1
                    diffs.append(abs(mean_a[x["key"]] - x["mean"]))
        out[sc] = {
            "seeds": len(seeds),
            "same_round1_training_set": same_set,
            "round1_shared_picks": shared,
            "round1_max_abs_mean_diff_K": max(diffs) if diffs else None,
        }
    return out


def novelty(runs, scns) -> pd.DataFrame:
    """For every target found: distance to the nearest material labeled before its round."""
    fr, noox = {}, {}
    for sc, scn in scns.items():
        X = scn.X.to_numpy(float)
        fr[sc] = dict(zip(scn.keys, X, strict=True))
        no_o = scn.X.assign(O=0.0)
        noox[sc] = dict(zip(scn.keys, data.composition_key(no_o).to_numpy(), strict=True))
    rows = []
    for rec in runs:
        sc = rec["scenario"]
        labeled = list(rec["initial"])
        for r in rec["rounds"]:
            lab_X = np.stack([fr[sc][k] for k in labeled])
            lab_no = {noox[sc][k] for k in labeled}
            for s in r["selected"]:
                if s["is_target"]:
                    d = float(np.abs(lab_X - fr[sc][s["key"]]).sum(1).min())
                    rows.append(
                        {
                            "phase": rec["phase"],
                            "scenario": sc,
                            "acquisition": rec["acquisition"],
                            "seed": rec["seed"],
                            "round": r["round"],
                            "family": s["family"],
                            "nn_l1": d,
                            "oxygen_variant_of_labeled": noox[sc][s["key"]] in lab_no,
                        }
                    )
            labeled += [s["key"] for s in r["selected"]]
    found = pd.DataFrame(rows)
    g = found.groupby(["phase", "scenario", "acquisition"])
    table = pd.DataFrame(
        {
            "targets_found": g.size(),
            "share_within_l1_0.01": g.nn_l1.apply(lambda d: float((d < 0.01).mean())),
            "share_within_l1_0.05": g.nn_l1.apply(lambda d: float((d < 0.05).mean())),
            "share_beyond_l1_0.1": g.nn_l1.apply(lambda d: float((d >= 0.1).mean())),
            "share_oxygen_variant_of_labeled": g.oxygen_variant_of_labeled.mean(),
            "median_nn_l1": g.nn_l1.median(),
        }
    ).reset_index()
    fam = found.groupby(["phase", "scenario", "acquisition", "family"]).size()
    return table, fam.rename("targets_found").reset_index()


def reliability(runs) -> pd.DataFrame:
    acc = {}
    for rec in runs:
        for r in rec["rounds"]:
            for b in r["pool"].get("p_top1_bins", []):
                k = (rec["phase"], rec["scenario"], rec["acquisition"], b["lo"], b["hi"])
                a = acc.setdefault(k, [0, 0.0, 0])
                a[0] += b["n"]
                a[1] += b["sum_p"]
                a[2] += b["hits"]
    rows = [
        {
            "phase": k[0],
            "scenario": k[1],
            "acquisition": k[2],
            "p_lo": k[3],
            "p_hi": k[4],
            "n": v[0],
            "predicted": v[1] / v[0] if v[0] else np.nan,
            "observed": v[2] / v[0] if v[0] else np.nan,
        }
        for k, v in acc.items()
    ]
    return pd.DataFrame(rows)


def batch_calibration(runs) -> pd.DataFrame:
    rows = []
    for rec in runs:
        if rec["acquisition"] not in discovery.TABPFN_ACQUISITIONS:
            continue
        for r in rec["rounds"]:
            sel = r["selected"]
            rows.append(
                {
                    "phase": rec["phase"],
                    "scenario": rec["scenario"],
                    "acquisition": rec["acquisition"],
                    "seed": rec["seed"],
                    "round": r["round"],
                    "expected_targets": sum(s.get("p_top1", np.nan) for s in sel),
                    "found_targets": sum(s["is_target"] for s in sel),
                    "expected_above_77K": sum(s.get("p_77K", np.nan) for s in sel),
                    "found_above_77K": sum(s["tc"] > 77 for s in sel),
                }
            )
    per_round = pd.DataFrame(rows)
    g = per_round.groupby(["phase", "scenario", "acquisition"])
    return (
        g[["expected_targets", "found_targets", "expected_above_77K", "found_above_77K"]]
        .sum()
        .reset_index()
    )


def timing(runs) -> dict:
    out = {}
    for rec in runs:
        if rec["acquisition"] in discovery.TABPFN_ACQUISITIONS:
            sec = [
                r["request"].get("seconds") for r in rec["rounds"] if r["request"].get("seconds")
            ]
            out.setdefault(rec["scenario"], []).extend(sec)
    return {
        sc: {
            "requests": len(v),
            "median_s": float(np.median(v)),
            "p95_s": float(np.percentile(v, 95)),
            "total_h": float(np.sum(v) / 3600),
        }
        for sc, v in out.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    scns = {sc: discovery.load_scenario(sc) for sc in discovery.SCENARIOS}
    runs = load_runs()
    if args.check:
        problems = [p for rec in runs for p in discovery.check_run(rec, scns[rec["scenario"]])]
        if problems:
            raise SystemExit(f"{len(problems)} integrity problems, first: {problems[0]}")
    curves(runs, scns).to_csv(OUT / "curves.csv", index=False, float_format="%.4f")
    summary(runs, scns).to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    paired(runs).to_csv(OUT / "paired.csv", index=False, float_format="%.4f")
    seeds = paired_seeds(runs)
    seeds.to_csv(OUT / "paired_seeds.csv", index=False)
    stall, by_stall = stalls(seeds)
    stall.to_csv(OUT / "stalls.csv", index=False)
    by_stall.to_csv(OUT / "paired_by_stall.csv", index=False, float_format="%.2f")
    (OUT / "ablation_check.json").write_text(json.dumps(ablation_check(runs), indent=1) + "\n")
    nov, fam = novelty(runs, scns)
    nov.to_csv(OUT / "novelty.csv", index=False, float_format="%.4f")
    fam.to_csv(OUT / "families.csv", index=False)
    reliability(runs).to_csv(OUT / "reliability.csv", index=False, float_format="%.5f")
    batch_calibration(runs).to_csv(OUT / "batch_calibration.csv", index=False, float_format="%.3f")
    (OUT / "timing.json").write_text(json.dumps(timing(runs), indent=1) + "\n")
    print(f"{len(runs)} runs evaluated")


if __name__ == "__main__":
    main()
