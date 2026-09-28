"""Phase 1 figures, drawn from results/01_audit/ and the raw data (no API calls).

Writes results/01_audit/figures/*.png. Run after experiments.01_data_audit:
`uv run python -m experiments.01_figures`.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data, plots
from src.plots import FAMILY_COLORS, FAMILY_LABELS, INK, INK_2, SPLIT_COLORS, SURFACE

AUDIT = data.ROOT / "results" / "01_audit"
FIG_DIR = AUDIT / "figures"
FAMILIES = (data.CUPRATE, data.IRON_BASED, data.OTHER)


def title(fig, text: str, sub: str) -> None:
    fig.suptitle(text, x=0.01, y=1.0, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.955, sub, ha="left", va="top", fontsize=8.5, color=INK_2)


def reference_line(ax, x: float, label: str, y: float = 0.97) -> None:
    ax.axvline(x, color=INK_2, linewidth=0.8)
    ax.text(
        x,
        y,
        f" {label}",
        transform=ax.get_xaxis_transform(),
        color=INK_2,
        fontsize=8,
        va="top",
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0},
    )


def tc_by_family(um, fam):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
    bins = np.arange(0, 190, 2.5)
    for ax, name in zip(axes, FAMILIES, strict=True):
        tc = um.loc[fam == name, data.TARGET]
        ax.hist(tc, bins=bins, color=FAMILY_COLORS[name], edgecolor=SURFACE, linewidth=0.6)
        ax.set_title(f"{FAMILY_LABELS[name]} · {len(tc):,} rows", fontsize=9, pad=4)
        ax.set_ylabel("Rows")
        reference_line(ax, 77, "77 K")
    axes[-1].set_xlabel("Reported critical temperature Tc (K)")
    title(fig, "Reported Tc by family", "Each panel has its own y-scale; 2.5 K bins.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def duplicate_spread(um, key, fam, n_groups: int = 12):
    top = key.value_counts().head(n_groups)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    labels, seen = [], set()
    for i, (k, n) in enumerate(top.items()):
        rows = um[key == k]
        name = fam[rows.index[0]]
        y = i + rng.uniform(-0.22, 0.22, size=n)
        ax.scatter(
            rows[data.TARGET],
            y,
            s=18,
            color=FAMILY_COLORS[name],
            edgecolors=SURFACE,
            linewidths=0.8,
            alpha=0.85,
            zorder=3,
            label=FAMILY_LABELS[name] if name not in seen else None,
        )
        seen.add(name)
        med = rows[data.TARGET].median()
        ax.plot([med, med], [i - 0.32, i + 0.32], color=INK, linewidth=2, zorder=4)
        labels.append(f"{rows[data.FORMULA].iloc[0]}  (n={n})")
    ax.set_yticks(range(len(top)), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Reported Tc (K); black tick = median")
    ax.legend(loc="lower right", handletextpad=0.3)
    title(
        fig,
        "Same composition, different reported Tc",
        f"The {n_groups} most repeated compositions; each dot is one row of the dataset.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def nn_distance_ecdf(nn, summary):
    example = summary["leakage"]["l1_distance_example"]["YBa2Cu3O7_vs_YBa2Cu3O6.9"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    # (x, vertical offset, alignment) per label: above the random curve, below-right
    # of the grouped curve, where the two are furthest apart.
    placement = {"random": (2.5e-3, 0.05, "center"), "grouped": (3e-3, -0.08, "left")}
    for kind in ("random", "grouped"):
        d = np.sort(nn.loc[nn.split_kind == kind, "nn_distance"].to_numpy())
        y = np.arange(1, len(d) + 1) / len(d)
        ax.step(d, y, where="post", color=SPLIT_COLORS[kind], label=f"{kind} split")
        x_at, dy, ha = placement[kind]
        y_at = np.searchsorted(d, x_at, side="right") / len(d)
        ax.text(x_at, y_at + dy, kind, color=INK_2, fontsize=8.5, ha=ha, va="center")
    ax.set_xscale("symlog", linthresh=1e-3, linscale=0.4)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.02)
    reference_line(ax, example, "YBa₂Cu₃O₇ vs YBa₂Cu₃O₆.₉", y=0.12)
    exact = summary["leakage"]["random"]["exact_duplicate_in_train"]
    ax.annotate(
        f"exact duplicate in train\n({exact:.0%} of random-split test rows,\n25-split mean)",
        xy=(0, 0.35),
        xytext=(2e-4, 0.55),
        color=INK_2,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8},
    )
    ax.set_xlabel("Distance to the nearest training composition (L1 over element fractions)")
    ax.set_ylabel("Share of test rows at or below")
    ax.legend(loc="lower right")
    title(
        fig,
        "How close is the nearest training material?",
        "Test rows of split 0 of each kind; x-axis linear below 0.001, log above.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def discovery_pool(pool, summary):
    dp = summary["discovery_pool"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    bins = np.arange(0, 150, 2.5)
    panels = (
        ("all", pool, FAMILIES, "All materials"),
        ("non_cuprate", pool[pool.family != data.CUPRATE], FAMILIES[1:], "Non-cuprates only"),
    )
    for ax, (name, sub, fams, heading) in zip(axes, panels, strict=True):
        stats = dp[name]
        ax.hist(
            [sub.loc[sub.family == f, "tc_median"] for f in fams],
            bins=bins,
            stacked=True,
            color=[FAMILY_COLORS[f] for f in fams],
            label=[FAMILY_LABELS[f] for f in fams],
            edgecolor=SURFACE,
            linewidth=0.6,
        )
        ax.set_yscale("log")
        ax.set_ylabel("Materials")
        ax.set_title(f"{heading} · {stats['materials']:,} materials", fontsize=9, pad=4)
        top = stats["top_1pct"]
        reference_line(ax, 77, f"77 K: {stats['above_77K']['hits']:,} materials", y=0.97)
        label = (
            f"top 1%: ≥ {top['threshold_K']:g} K, {top['hits']} materials,\n"
            f" random search ≈ {top['random_expected_tries_to_first_hit']:.0f} tries to first hit"
        )
        if name == "non_cuprate":
            clean = dp["non_cuprate_excluding_suspects"]["top_1pct"]
            label += (
                f"\n without the {len(dp['suspect_materials'])} flagged entries: "
                f"≥ {clean['threshold_K']:g} K, {clean['hits']} materials"
            )
        reference_line(ax, top["threshold_K"], label, y=0.80)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in FAMILIES]
    fig.legend(
        handles,
        [FAMILY_LABELS[f] for f in FAMILIES],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.925),
        ncol=3,
        handlelength=1.0,
        columnspacing=1.2,
    )
    axes[-1].set_xlabel("Median reported Tc per material (K)")
    title(
        fig,
        "Discovery pool: one row per composition",
        "Log y-scale, stacked by family. Targets for the discovery loop in Phase 4.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def thermal_conductivity(train, fam, summary):
    modes = summary["features"]["range_ThermalConductivity_mode_by_family"]
    rtc = train["range_ThermalConductivity"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.2), sharex=True)
    bins = np.arange(0, 440, 5)
    for ax, name in zip(axes, FAMILIES, strict=True):
        x = rtc[fam == name]
        ax.hist(x, bins=bins, color=FAMILY_COLORS[name], edgecolor=SURFACE, linewidth=0.6)
        m = modes[name]
        ax.set_title(
            f"{FAMILY_LABELS[name]} · most common value {m['value']:g} ({m['share']:.1%} of rows)",
            fontsize=9,
            pad=4,
        )
        ax.set_ylabel("Rows")
    axes[-1].set_xlabel("range_ThermalConductivity (W/(m·K))")
    title(fig, "The paper's top feature by family", "Each panel has its own y-scale; 5-unit bins.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main():
    plots.setup()
    summary = json.loads((AUDIT / "summary.json").read_text())
    train, um = data.load_train(), data.load_unique_m()
    key, fam = data.composition_key(um), data.family(um)
    pool = pd.read_csv(AUDIT / "pool.csv")
    nn = pd.read_csv(AUDIT / "nn_split0.csv")

    figures = {
        "tc_by_family": tc_by_family(um, fam),
        "duplicate_spread": duplicate_spread(um, key, fam),
        "nn_distance_ecdf": nn_distance_ecdf(nn, summary),
        "discovery_pool": discovery_pool(pool, summary),
        "range_thermal_conductivity": thermal_conductivity(train, fam, summary),
    }
    for name, fig in figures.items():
        plots.save(fig, FIG_DIR / f"{name}.png")
        print(f"Wrote {FIG_DIR / name}.png")


if __name__ == "__main__":
    main()
