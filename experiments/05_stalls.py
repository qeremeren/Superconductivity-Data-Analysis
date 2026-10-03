"""Phase 5: why greedy search stalls on some start sets (from the committed run records only).

Descriptive and post hoc. "Stall seeds" are the start sets where both greedy methods stalled
(discovery.STALL_MAX): main 0, 4, 8; hard 0-3. Target chemistry = what most targets are (main:
contains Hg or Tl; hard: iron-based), checked below. Writes results/05_why/:
  stall_runs.csv          per run (main runs, all methods): targets found, stalled, best Tc
                          after rounds 1/5/10/20, share of picks in the target chemistry,
                          median Tc of the picks, best Tc in the start set
  stall_trajectories.csv  best Tc so far after every round, per run
  stall_round1.csv        EI vs greedy TabPFN in round 1 (same fit, only the rule differs): the
                          batch's mean predicted Tc, upper spread (q90 - mean), P(top 1%),
                          realised Tc, targets, share in the target chemistry
  stall_summary.json      target chemistry shares; most-picked chemical systems (elements
                          other than O) of stalled greedy runs and of EI on the same seeds;
                          summaries by method and stall; round-1 paired differences
  figures/stalls.png      best Tc so far per round on the stall seeds, every method

Run from the repo root: `uv run python -m experiments.05_stalls`.
"""

from __future__ import annotations

import collections
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, discovery, plots
from src.plots import INK, INK_2

OUT = config.RESULTS_DIR / "05_why"
RUNS = config.RESULTS_DIR / "04_discovery" / "runs" / "main"
METHODS = ("ei", "greedy_tabpfn", "greedy_xgb", "gp_ei")
GREEDY = ("greedy_tabpfn", "greedy_xgb")
TARGET_CHEMISTRY = {"main": "contains Hg or Tl", "hard": "iron-based"}
COLORS = {"ei": "#e34948", "greedy_tabpfn": "#4a3aa7", "greedy_xgb": "#eda100", "gp_ei": "#0a8fa3"}
STYLE = {"gp_ei": "--"}
LABELS = {
    "ei": "TabPFN EI",
    "greedy_tabpfn": "TabPFN greedy",
    "greedy_xgb": "XGBoost greedy",
    "gp_ei": "GP EI",
}


def target_chemistry(scn) -> np.ndarray:
    if scn.name == "main":
        return ((scn.X["Hg"] > 0) | (scn.X["Tl"] > 0)).to_numpy()
    return scn.family == "iron-based"


def system(scn, i) -> str:
    row = scn.X.iloc[i]
    return "-".join(sorted(c for c in scn.X.columns if row[c] > 0 and c != "O"))


def load(sc, acq, seed) -> dict:
    return json.loads((RUNS / sc / acq / f"seed_{seed:03d}.json").read_text())


def picks_of(rec, idx) -> np.ndarray:
    return np.array([idx[s["key"]] for r in rec["rounds"] for s in r["selected"]])


def per_run(sc, scn, chem):
    idx = {k: i for i, k in enumerate(scn.keys)}
    runs, traj = [], []
    for acq in METHODS:
        for seed in discovery.MAIN_SEEDS:
            rec = load(sc, acq, seed)
            init = np.array([idx[k] for k in rec["initial"]])
            picks = picks_of(rec, idx)
            found = rec["rounds"][-1]["found_cumulative"]
            best = np.maximum.accumulate(
                [
                    max(scn.tc[init].max(), max(s["tc"] for s in r["selected"]))
                    for r in rec["rounds"]
                ]
            )
            for r, b in enumerate(best, start=1):
                traj.append(
                    {"scenario": sc, "acquisition": acq, "seed": seed, "round": r, "best_tc": b}
                )
            runs.append(
                {
                    "scenario": sc,
                    "acquisition": acq,
                    "seed": seed,
                    "found": found,
                    "stalled": found <= discovery.STALL_MAX,
                    **{f"best_tc_round_{r}": float(best[r - 1]) for r in (1, 5, 10, 20)},
                    "share_picks_target_chem": float(chem[picks].mean()),
                    "median_pick_tc": float(np.median(scn.tc[picks])),
                    "mean_pick_tc": float(scn.tc[picks].mean()),
                    "mean_pick_pred": float(
                        np.mean([s["mean"] for r in rec["rounds"] for s in r["selected"]])
                    )
                    if acq != "random"
                    else np.nan,
                    "start_max_tc": float(scn.tc[init].max()),
                }
            )
    return runs, traj


def round1(sc, scn, chem) -> list[dict]:
    idx = {k: i for i, k in enumerate(scn.keys)}
    rows = []
    for seed in discovery.MAIN_SEEDS:
        for acq in ("ei", "greedy_tabpfn"):
            sel = load(sc, acq, seed)["rounds"][0]["selected"]
            i = np.array([idx[s["key"]] for s in sel])
            rows.append(
                {
                    "scenario": sc,
                    "seed": seed,
                    "acquisition": acq,
                    "pred_mean": float(np.mean([s["mean"] for s in sel])),
                    "pred_upper_spread": float(np.mean([s["q90"] - s["mean"] for s in sel])),
                    "p_top1": float(np.mean([s["p_top1"] for s in sel])),
                    "realised_tc": float(scn.tc[i].mean()),
                    "targets": int(scn.is_target[i].sum()),
                    "share_target_chem": float(chem[i].mean()),
                }
            )
    return rows


def top_systems(scn, recs, n=5) -> list:
    idx = {k: i for i, k in enumerate(scn.keys)}
    picks = np.concatenate([picks_of(r, idx) for r in recs])
    count = collections.Counter(system(scn, i) for i in picks)
    return [[s, c, round(c / len(picks), 3)] for s, c in count.most_common(n)]


def figure(traj, stall_seeds, thresholds):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    for ax, sc in zip(axes, discovery.SCENARIOS, strict=True):
        d = traj[(traj.scenario == sc) & traj.seed.isin(stall_seeds[sc])]
        for acq in ("gp_ei", "greedy_xgb", "greedy_tabpfn", "ei"):
            for k, (_, s) in enumerate(d[d.acquisition == acq].groupby("seed")):
                ax.plot(
                    s["round"] * discovery.BATCH,
                    s.best_tc,
                    color=COLORS[acq],
                    linestyle=STYLE.get(acq, "-"),
                    linewidth=1.4,
                    alpha=0.85,
                    label=LABELS[acq] if k == 0 else None,
                )
        ax.axhline(thresholds[sc], color=INK_2, linewidth=0.8)
        ax.set_xlabel("Experiments")
        ax.set_xlim(0, 205)
        seeds = ", ".join(map(str, stall_seeds[sc]))
        ax.set_title(
            f"{sc.capitalize()} pool, seeds {seeds} (line: target, Tc ≥ {thresholds[sc]:.0f} K)",
            fontsize=9,
            pad=4,
        )
    axes[0].set_ylabel("Best Tc measured so far (K)")
    axes[1].legend(loc="upper left", fontsize=7.5)
    h = fig.get_figheight()
    fig.suptitle(
        "Where greedy search stalled, it plateaued just below the target",
        x=0.01,
        y=1.0,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        1 - 0.32 / h,
        "The start sets on which both greedy methods stalled; one line per run.",
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / h))
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, traj, r1, summary, thresholds, stall_seeds = [], [], [], {}, {}, {}
    for sc in discovery.SCENARIOS:
        scn = discovery.load_scenario(sc)
        chem = target_chemistry(scn)
        rr, tt = per_run(sc, scn, chem)
        runs += rr
        traj += tt
        r1 += round1(sc, scn, chem)
        thresholds[sc] = scn.threshold
        df = pd.DataFrame(rr)
        both = df[df.acquisition.isin(GREEDY)].groupby("seed").stalled.all()
        stall_seeds[sc] = [int(s) for s in both[both].index]
        stalled_greedy = [load(sc, a, s) for a in GREEDY for s in stall_seeds[sc]]
        ei_same = [load(sc, "ei", s) for s in stall_seeds[sc]]
        summary[sc] = {
            "target_chemistry": TARGET_CHEMISTRY[sc],
            "share_targets_in_target_chemistry": float(chem[scn.is_target].mean()),
            "stall_seeds_both_greedy": stall_seeds[sc],
            "top_systems_stalled_greedy_picks": top_systems(scn, stalled_greedy),
            "top_systems_ei_picks_same_seeds": top_systems(scn, ei_same),
        }
    runs, traj, r1 = pd.DataFrame(runs), pd.DataFrame(traj), pd.DataFrame(r1)
    summary["by_method_and_stall"] = [
        {"scenario": k[0], "acquisition": k[1], "stalled": bool(k[2]), "runs": int(len(v))}
        | {
            c: float(v[c].mean())
            for c in (
                "mean_pick_tc",
                "mean_pick_pred",
                "best_tc_round_5",
                "best_tc_round_20",
                "share_picks_target_chem",
                "median_pick_tc",
                "start_max_tc",
            )
        }
        | {"best_tc_round_20_max": float(v.best_tc_round_20.max())}
        for k, v in runs.groupby(["scenario", "acquisition", "stalled"])
    ]
    summary["ei_on_stall_seeds"] = {
        sc: {
            c: float(
                runs[
                    (runs.scenario == sc)
                    & (runs.acquisition == "ei")
                    & runs.seed.isin(stall_seeds[sc])
                ][c].mean()
            )
            for c in ("found", "best_tc_round_5", "best_tc_round_10", "share_picks_target_chem")
        }
        for sc in discovery.SCENARIOS
    }
    pair = r1.pivot_table(index=["scenario", "seed"], columns="acquisition")
    summary["round1_ei_minus_greedy_tabpfn_mean_over_seeds"] = {
        sc: {
            c: float((pair.loc[sc][(c, "ei")] - pair.loc[sc][(c, "greedy_tabpfn")]).mean())
            for c in (
                "pred_mean",
                "pred_upper_spread",
                "p_top1",
                "realised_tc",
                "targets",
                "share_target_chem",
            )
        }
        for sc in discovery.SCENARIOS
    }
    runs.to_csv(OUT / "stall_runs.csv", index=False, float_format="%.4f")
    traj.to_csv(OUT / "stall_trajectories.csv", index=False, float_format="%.2f")
    r1.to_csv(OUT / "stall_round1.csv", index=False, float_format="%.4f")
    (OUT / "stall_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    plots.setup()
    plots.save(figure(traj, stall_seeds, thresholds), OUT / "figures" / "stalls.png")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_method_and_stall"}, indent=1))
    print(pd.DataFrame(summary["by_method_and_stall"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
