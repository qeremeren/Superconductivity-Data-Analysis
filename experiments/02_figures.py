"""Phase 2 figures from results/02_benchmark/ (no API calls, no model fitting).

Writes results/02_benchmark/figures/:
  paired_difference_grouped.png  per-split RMSE of TabPFN minus tuned XGBoost, grouped split
  rmse_by_split_kind.png         RMSE per model and feature set on the three repeated splits
  grouped_subgroups.png          grouped-split RMSE by Tc band, family and missing-oxygen rows
Run after experiments.02_metrics: `uv run python -m experiments.02_figures`.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import benchmark, plots
from src.plots import INK, INK_2, MODEL_COLORS, SPLIT_COLORS, SURFACE

FIG_DIR = benchmark.OUT / "figures"
LABELS = {
    "tabpfn": "TabPFN-3.5",
    "xgb_tuned": "XGBoost tuned",
    "xgb_hamidieh": "XGBoost published",
    "nn_lookup": "1-NN lookup",
    "linear": "Linear",
}
KIND_LABELS = {"random": "random", "grouped": "grouped", "grouped_no_oxygen": "grouped, no oxygen"}


def title(fig, text: str, sub: str) -> float:
    """Title and subtitle at fixed distances from the top; returns the layout top."""
    h = fig.get_figheight()
    fig.suptitle(text, x=0.01, y=1.0, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.32 / h, sub, ha="left", va="top", fontsize=8.5, color=INK_2)
    return 1 - 0.6 / h


def paired_difference(per_split: pd.DataFrame, paired: list[dict]):
    ps = per_split[(per_split.group == "all") & (per_split.kind == "grouped")]
    rmse = ps.pivot_table(index=["features", "split"], columns="model", values="rmse")
    diff = (rmse["tabpfn"] - rmse["xgb_tuned"]).rename("diff").reset_index()
    stats = {p["features"]: p for p in paired if p["kind"] == "grouped" and p["b"] == "xgb_tuned"}
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    rng = np.random.default_rng(0)
    order = ["composition", "engineered"]
    for i, fs in enumerate(order):
        d = diff[diff.features == fs]["diff"].to_numpy()
        ax.scatter(
            d,
            i + rng.uniform(-0.12, 0.12, len(d)),
            s=30,
            color=SPLIT_COLORS["grouped"],
            edgecolors=SURFACE,
            linewidths=1,
            zorder=3,
        )
        s = stats[fs]
        ax.plot([s["mean_diff"]] * 2, [i - 0.25, i + 0.25], color=INK, linewidth=2, zorder=4)
        ax.text(
            d.min(),
            i - 0.3,
            f"mean {s['mean_diff']:+.2f} K · TabPFN better on "
            f"{s['a_better_splits']}/{s['n_splits']} splits",
            color=INK_2,
            fontsize=8,
            va="bottom",
        )
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(range(len(order)), [f"{fs} features" for fs in order])
    ax.set_ylim(-0.75, len(order) - 0.35)
    ax.set_xlim(min(diff["diff"].min() - 0.2, -0.1), 0.15)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(
        "Per-split RMSE difference, TabPFN minus tuned XGBoost (K); below 0 = TabPFN better"
    )
    top = title(
        fig,
        "TabPFN-3.5 vs tuned XGBoost on the grouped split",
        "Each dot is one of 25 grouped splits; both models see the same splits, seeds and "
        "features. Black tick = mean.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def rmse_by_split_kind(table: pd.DataFrame):
    t = table[(table.group == "all") & (table.kind.isin(KIND_LABELS)) & (table.model != "linear")]
    rows = [
        ("tabpfn", "composition"),
        ("tabpfn", "engineered"),
        ("xgb_tuned", "engineered"),
        ("xgb_tuned", "composition"),
        ("xgb_hamidieh", "engineered"),
        ("xgb_hamidieh", "composition"),
        ("nn_lookup", "composition"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for kind in KIND_LABELS:
        sub = t[t.kind == kind].set_index(["model", "features"])
        x = [sub.loc[r, "rmse"] for r in rows]
        ax.scatter(
            x,
            range(len(rows)),
            s=42,
            color=SPLIT_COLORS[kind],
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=3,
            label=KIND_LABELS[kind],
        )
    ax.set_yticks(range(len(rows)), [f"{LABELS[m]} · {fs}" for m, fs in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMSE over 25 splits, sqrt(mean MSE) (K)")
    ax.legend(loc="upper right", title="split", title_fontsize=8)
    top = title(
        fig,
        "RMSE by model and split",
        "Lower is better. Linear regression (17.6-19.8 K) is left off the axis; see metrics.csv.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def grouped_subgroups(table: pd.DataFrame):
    order = [
        "all",
        "Tc < 10 K",
        "10-77 K",
        "Tc > 77 K",
        "family: cuprate",
        "family: iron-based",
        "family: other",
        "oxygen amount missing",
    ]
    t = table[table.kind == "grouped"].set_index(["model", "features", "group"]).sort_index()
    pairs = {"tabpfn": ("tabpfn", "composition"), "xgb_tuned": ("xgb_tuned", "engineered")}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, g in enumerate(order):
        x = {k: t.loc[(*v, g), "rmse"] for k, v in pairs.items()}
        ax.plot([x["tabpfn"], x["xgb_tuned"]], [i, i], color=plots.GRID, linewidth=2, zorder=2)
        for k in pairs:
            ax.scatter(
                x[k],
                i,
                s=42,
                color=MODEL_COLORS[k],
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=3,
                label=f"{LABELS[k]} · {pairs[k][1]}" if i == 0 else None,
            )
        ax.text(
            max(x.values()) + 0.25,
            i,
            f"{x['tabpfn']:.1f} vs {x['xgb_tuned']:.1f} K",
            va="center",
            fontsize=7.5,
            color=INK_2,
        )
    rows_per_split = t.loc[("tabpfn", "composition"), "rows_per_split"].reindex(order)
    ax.set_yticks(range(len(order)), [f"{g} (n={n:,.0f})" for g, n in rows_per_split.items()])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=ax.get_xlim()[1] + 2.5)
    ax.set_xlabel("RMSE over 25 grouped splits (K); n = test rows per split")
    ax.legend(loc="upper right")
    top = title(
        fig,
        "Grouped split: error by Tc band, family and missing-oxygen rows",
        "Each model on its better feature set for the grouped split.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def main():
    plots.setup()
    per_split = pd.read_parquet(benchmark.OUT / "per_split_metrics.parquet")
    table = pd.read_csv(benchmark.OUT / "metrics.csv")
    paired = json.loads((benchmark.OUT / "paired.json").read_text())
    figures = {
        "paired_difference_grouped": paired_difference(per_split, paired),
        "rmse_by_split_kind": rmse_by_split_kind(table),
        "grouped_subgroups": grouped_subgroups(table),
    }
    for name, fig in figures.items():
        plots.save(fig, FIG_DIR / f"{name}.png")
        print(f"Wrote {FIG_DIR / name}.png")


if __name__ == "__main__":
    main()
