"""Phase 4 figures from results/04_discovery/ (no API calls). Run after experiments.04_evaluate.

Writes results/04_discovery/figures/:
  discovery_curves.png  targets found vs experiments, main runs, 95% band over 10 seeds
  first_hit.png         tries to the first target per seed (runs without a hit marked)
  novelty.png           how close each found target was to an already labeled material
  reliability.png       predicted P(top 1%) vs observed share over the pool: TabPFN (EI runs)
                        vs GP (GP EI runs)
  per_seed.png          targets found per seed: EI vs greedy TabPFN (same model, with vs without
                        uncertainty) and greedy XGBoost, same start set on each row
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, discovery, plots
from src.plots import INK, INK_2, SURFACE

OUT = config.RESULTS_DIR / "04_discovery"
FIG = OUT / "figures"
# EI red, greedy TabPFN violet, greedy XGBoost yellow: validated all-pairs (yellow below 3:1
# contrast, so always labeled). GP EI teal, chosen by simulated deutan/protan colour distance
# to the other three (min dE76 39) with 3.7:1 contrast, and always dashed / diamond as well.
# Random is a reference line in ink, not a series colour.
COLORS = {"ei": "#e34948", "greedy_tabpfn": "#4a3aa7", "greedy_xgb": "#eda100", "gp_ei": "#0a8fa3"}
STYLE = {"gp_ei": "--"}
MARKER = {"gp_ei": "D"}
METHODS = ("ei", "greedy_tabpfn", "greedy_xgb", "gp_ei")
LABELS = {
    "ei": "TabPFN EI",
    "greedy_tabpfn": "TabPFN greedy (mean)",
    "greedy_xgb": "XGBoost greedy",
    "gp_ei": "GP EI (Phase 5 baseline)",
    "random": "random search",
}
TITLES = {
    "main": "Main: top 1% of all materials (Tc ≥ 119 K)",
    "hard": "Hard: top 1% of non-cuprates (Tc ≥ 44 K)",
}


def title(fig, text, sub):
    h = fig.get_figheight()
    fig.suptitle(text, x=0.01, y=1.0, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.32 / h, sub, ha="left", va="top", fontsize=8.5, color=INK_2)
    return 1 - 0.6 / h


def discovery_curves(curves):
    c = curves[curves.phase == "main"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    for ax, sc in zip(axes, discovery.SCENARIOS, strict=True):
        d = c[c.scenario == sc]
        for acq in METHODS:
            a = d[d.acquisition == acq].sort_values("experiments")
            ax.fill_between(
                a.experiments,
                a.found_lo.clip(lower=0),
                a.found_hi,
                color=COLORS[acq],
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                a.experiments,
                a.found_mean,
                color=COLORS[acq],
                linestyle=STYLE.get(acq, "-"),
                label=LABELS[acq],
            )
            ax.text(
                a.experiments.iloc[-1] + 4,
                a.found_mean.iloc[-1],
                f"{a.found_mean.iloc[-1]:.0f}",
                va="center",
                fontsize=7.5,
                color=INK_2,
            )
        r = d[d.acquisition == "random"].sort_values("experiments")
        ax.plot(r.experiments, r.random_expected, color=INK_2, linewidth=1, label="random (exact)")
        ax.set_title(TITLES[sc], fontsize=9, pad=4)
        ax.set_xlabel("Experiments (materials measured after the 50 starting ones)")
        ax.set_xlim(0, 225)
    axes[0].set_ylabel("Distinct targets found (mean of 10 seeds)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    top = title(
        fig,
        "Simulated discovery: targets found per experiment",
        "Each round: fit on what is measured, measure the 10 best-scored materials. "
        "Band = 95% interval over seeds.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def first_hit(summary_runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharex=True)
    order = [*METHODS, "random"]
    for ax, sc in zip(axes, discovery.SCENARIOS, strict=True):
        d = summary_runs[summary_runs.scenario == sc]
        for i, acq in enumerate(order):
            v = d[d.acquisition == acq].first_hit
            hit = v.dropna().to_numpy()
            color = COLORS.get(acq, INK_2)
            ax.scatter(
                hit,
                np.full(len(hit), i),
                marker=MARKER.get(acq, "o"),
                s=26,
                color=color,
                edgecolors=SURFACE,
                linewidths=1,
                zorder=3,
            )
            miss = int(v.isna().sum())
            if miss:
                ax.scatter([212], [i], s=30, marker="x", color=INK_2, zorder=3)
                ax.text(219, i, f"{miss} no hit", va="center", fontsize=7.5, color=INK_2)
        exp = d.random_expected_first_hit.iloc[0]
        ax.axvline(exp, color=INK_2, linewidth=0.8)
        ax.text(exp + 3, len(order) - 1.5, f"random: {exp:.0f} expected", fontsize=7.5, color=INK_2)
        ax.set_yticks(range(len(order)), [LABELS[a] for a in order])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 250)
        ax.set_title(TITLES[sc], fontsize=9, pad=4)
        ax.set_xlabel("Experiments to the first target")
    top = title(
        fig,
        "How soon is the first target found?",
        "One dot per seed (10 seeds); x = runs that found none in 200 experiments.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def novelty(nov):
    d = nov[nov.phase == "main"].copy()
    d["< 0.01"] = d["share_within_l1_0.01"]
    d["0.01-0.05"] = d["share_within_l1_0.05"] - d["share_within_l1_0.01"]
    d["0.05-0.1"] = 1 - d["share_within_l1_0.05"] - d["share_beyond_l1_0.1"]
    d["≥ 0.1"] = d["share_beyond_l1_0.1"]
    bins = ["< 0.01", "0.01-0.05", "0.05-0.1", "≥ 0.1"]
    ramp = ["#104281", "#2a78d6", "#6da7ec", "#b7d3f6"]  # sequential blue, dark = closest
    order = [*METHODS, "random"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for ax, sc in zip(axes, discovery.SCENARIOS, strict=True):
        s = d[d.scenario == sc].set_index("acquisition").reindex(order)
        left = np.zeros(len(order))
        for b, col in zip(bins, ramp, strict=True):
            ax.barh(
                range(len(order)),
                s[b],
                left=left,
                color=col,
                edgecolor=SURFACE,
                linewidth=1.5,
                height=0.6,
                label=b,
            )
            left += s[b].to_numpy()
        for i, n in enumerate(s.targets_found):
            ax.text(1.01, i, f"n={n:.0f}", va="center", fontsize=7.5, color=INK_2)
        ax.set_xlim(0, 1.12)
        ax.set_title(TITLES[sc], fontsize=9, pad=4)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Share of found targets")
    axes[0].set_yticks(range(len(order)), [LABELS[a] for a in order])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, 0),
        title="L1 distance to the nearest already-measured material",
        title_fontsize=8,
    )
    top = title(
        fig,
        "How new were the targets found?",
        "All finds over 10 seeds. YBa₂Cu₃O₇ vs YBa₂Cu₃O₆.₉ is at distance 0.007.",
    )
    fig.tight_layout(rect=(0, 0.14, 1, top))
    return fig


def reliability(rel):
    r = rel[(rel.phase == "main") & (rel.n > 0) & (rel.observed > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
    for ax, sc in zip(axes, discovery.SCENARIOS, strict=True):
        ax.plot([1e-4, 1], [1e-4, 1], color=INK_2, linewidth=0.8)
        for acq in ("ei", "gp_ei"):
            d = r[(r.scenario == sc) & (r.acquisition == acq)]
            ax.plot(
                d.predicted,
                d.observed,
                color=COLORS[acq],
                linestyle=STYLE.get(acq, "-"),
                marker=MARKER.get(acq, "o"),
                markersize=5,
                markeredgecolor=SURFACE,
                label={"ei": "TabPFN (EI runs)", "gp_ei": "GP (GP EI runs)"}[acq],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TITLES[sc], fontsize=9, pad=4)
        ax.set_xlabel("Predicted P(top 1%), binned")
    axes[0].set_ylabel("Observed share that are targets")
    axes[0].legend(loc="upper left", fontsize=8)
    top = title(
        fig,
        "Are the predicted P(top 1%) reliable during discovery?",
        "Every round's unlabeled pool, 10 seeds. Log axes; diagonal = calibrated; "
        "bins with no target omitted.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def per_seed(seeds, stall_max):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
    offsets = {"ei": 0.0, "greedy_tabpfn": -0.24, "greedy_xgb": 0.16, "gp_ei": 0.32}
    for ax, sc in zip(axes, discovery.SCENARIOS, strict=True):
        d = seeds[seeds.scenario == sc].sort_values("seed")
        ax.axvspan(-5, stall_max + 0.5, color=INK_2, alpha=0.08, linewidth=0)
        ax.text(stall_max + 2, -0.9, f"stalled (≤ {stall_max})", fontsize=7.5, color=INK_2)
        for y, (_, r) in enumerate(d.iterrows()):
            ax.plot(
                [r.ei, r.greedy_tabpfn],
                [y, y + offsets["greedy_tabpfn"]],
                color=INK_2,
                linewidth=0.8,
                zorder=1,
            )
        for acq in ("gp_ei", "greedy_xgb", "greedy_tabpfn", "ei"):
            ax.scatter(
                d[acq],
                np.arange(len(d)) + offsets[acq],
                marker=MARKER.get(acq, "o"),
                s=30 if acq == "gp_ei" else 34,
                color=COLORS[acq],
                edgecolors=SURFACE,
                linewidths=1,
                zorder=3,
                label=LABELS[acq],
            )
        ax.set_yticks(range(len(d)), [f"seed {x}" for x in d.seed])
        ax.set_ylim(len(d) - 0.4, -1.3)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-5, 120)
        ax.set_title(TITLES[sc], fontsize=9, pad=4)
        ax.set_xlabel("Targets found after 200 experiments")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=4, fontsize=8)
    top = title(
        fig,
        "Same start, same model: EI vs greedy, seed by seed",
        "Each row: one seed, the same 50 starting materials for every method. "
        "Line: EI vs greedy TabPFN (same model and first fit).",
    )
    fig.tight_layout(rect=(0, 0.07, 1, top))
    return fig


def main():
    plots.setup()
    curves = pd.read_csv(OUT / "curves.csv")
    runs = []
    for path in sorted((OUT / "runs" / "main").glob("*/*/seed_*.json")):
        rec = json.loads(path.read_text())
        hit = None
        for r in rec["rounds"]:
            for j, s in enumerate(r["selected"]):
                if s["is_target"] and hit is None:
                    hit = (r["round"] - 1) * rec["batch"] + j + 1
        runs.append(
            {
                "scenario": rec["scenario"],
                "acquisition": rec["acquisition"],
                "seed": rec["seed"],
                "first_hit": hit,
                "random_expected_first_hit": (rec["pool_size"] - rec["n_initial"] + 1)
                / (rec["targets_in_pool"] + 1),
            }
        )
    figures = {
        "discovery_curves": discovery_curves(curves),
        "first_hit": first_hit(pd.DataFrame(runs)),
        "novelty": novelty(pd.read_csv(OUT / "novelty.csv")),
        "reliability": reliability(pd.read_csv(OUT / "reliability.csv")),
        "per_seed": per_seed(pd.read_csv(OUT / "paired_seeds.csv"), discovery.STALL_MAX),
    }
    for name, fig in figures.items():
        plots.save(fig, FIG / f"{name}.png")
        print(f"Wrote {FIG / name}.png")


if __name__ == "__main__":
    main()
