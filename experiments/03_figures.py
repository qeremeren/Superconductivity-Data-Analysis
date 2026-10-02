"""Phase 3 figures from results/03_learning_curves/ and results/03_uncertainty/ (no API/fitting).

Writes results/03_uncertainty/figures/:
  learning_curves.png       RMSE vs training rows, TabPFN vs tuned XGBoost, both feature sets
  calibration_curve.png     nominal vs empirical central-interval coverage, grouped split
  lfo_coverage.png          95% and 80% coverage and 95% width on each held-out family,
                            next to the same family in-distribution (grouped split)
  pit_hist.png              TabPFN PIT histogram, grouped split
  p77_reliability.png       TabPFN predicted P(Tc > 77 K) vs observed frequency, grouped split
Run after experiments.03_evaluate: `uv run python -m experiments.03_figures`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import benchmark, plots
from src.plots import INK, INK_2, SURFACE

RES = benchmark.config.RESULTS_DIR
LC, UNC = RES / "03_learning_curves", RES / "03_uncertainty"
FIG = UNC / "figures"
# TabPFN red and XGBoost yellow as in Phase 2; conformal violet. The three pass the palette
# validator all-pairs; yellow is below 3:1 contrast, so its series are always labeled.
COLORS = {
    "tabpfn": "#e34948",
    "xgb_tuned": "#eda100",
    "quantile_regression": "#eda100",
    "conformal": "#4a3aa7",
}
LABELS = {
    "tabpfn": "TabPFN-3.5",
    "xgb_tuned": "XGBoost tuned",
    "quantile_regression": "XGBoost quantile regression",
    "conformal": "XGBoost conformal",
}
FEATURES = "composition"  # TabPFN's better set on the grouped split; tables have both


def title(fig, text, sub):
    h = fig.get_figheight()
    fig.suptitle(text, x=0.01, y=1.0, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.32 / h, sub, ha="left", va="top", fontsize=8.5, color=INK_2)
    return 1 - 0.6 / h


def learning_curves(curves):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for ax, fs in zip(axes, ("composition", "engineered"), strict=True):
        for model in ("tabpfn", "xgb_tuned"):
            c = curves[(curves.model == model) & (curves.features == fs)].sort_values("n_train")
            if c.empty:
                continue
            ax.fill_between(
                c.n_train,
                c.rmse - c.rmse_sd,
                c.rmse + c.rmse_sd,
                color=COLORS[model],
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                c.n_train,
                c.rmse,
                color=COLORS[model],
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                label=LABELS[model],
            )
            last = c.iloc[-1]
            ax.text(
                last.n_train * 1.08,
                last.rmse,
                f"{last.rmse:.1f}",
                va="center",
                fontsize=7.5,
                color=INK_2,
            )
        ax.set_xscale("log")
        ax.set_title(f"{fs} features", fontsize=9, pad=4)
        ax.set_xlabel("Training rows (log scale; last point = full split)")
    axes[0].set_ylabel("Test RMSE (K), mean over 5 grouped splits")
    axes[0].legend(loc="upper right")
    top = title(
        fig,
        "Learning curves on the grouped split",
        "Same nested training subsets and the same test set for both models; band = "
        "±1 SD over splits.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def calibration_curve(curve, cal):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=0.8)
    ax.text(0.92, 0.95, "ideal", color=INK_2, fontsize=7.5, ha="right")
    for model in ("tabpfn", "quantile_regression", "conformal"):
        c = curve[
            (curve.model == model) & (curve.kind == "grouped") & (curve.features == FEATURES)
        ].sort_values("nominal")
        if c.empty:
            continue
        x = [*c.nominal, 0.95]
        row = cal[
            (cal.model == model)
            & (cal.kind == "grouped")
            & (cal.features == FEATURES)
            & (cal.group == "all")
        ]
        yv = [*c.empirical, float(row["coverage_95%"].iloc[0])]
        ax.plot(
            x,
            yv,
            color=COLORS[model],
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            label=LABELS[model],
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal central-interval level")
    ax.set_ylabel("Share of test rows inside the interval")
    ax.legend(loc="lower right")
    top = title(
        fig,
        "Interval calibration, grouped split",
        f"{FEATURES} features, all 25 splits pooled; above the line = conservative.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def lfo_coverage(lfo):
    d = lfo[lfo.features == FEATURES]
    models = [m for m in ("tabpfn", "quantile_regression", "conformal") if m in set(d.model)]
    families = ["cuprate", "iron-based", "other"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.8), sharey=True)
    panels = (
        ("coverage_95%", "95% interval coverage", 0.95),
        ("coverage_80%", "80% interval coverage", 0.80),
        ("width_95%", "95% interval width (K)", None),
    )
    for ax, (col, label, nominal) in zip(axes, panels, strict=True):
        for i, model in enumerate(models):
            m = d[d.model == model].set_index("held_out_family").reindex(families)
            y = np.arange(len(families)) + (i - (len(models) - 1) / 2) * 0.22
            ax.scatter(
                m[f"in_distribution_{col}"],
                y,
                s=30,
                facecolors=SURFACE,
                edgecolors=COLORS[model],
                linewidths=1.5,
                zorder=3,
            )
            ax.scatter(
                m[f"held_out_{col}"],
                y,
                s=36,
                color=COLORS[model],
                edgecolors=SURFACE,
                linewidths=1,
                zorder=4,
                label=LABELS[model],
            )
            for xi, xo, yi in zip(
                m[f"in_distribution_{col}"], m[f"held_out_{col}"], y, strict=True
            ):
                ax.plot([xi, xo], [yi, yi], color=plots.GRID, linewidth=1.5, zorder=2)
        if nominal is not None:
            ax.axvline(nominal, color=INK_2, linewidth=0.8)
            ax.set_xlim(0, 1.02)
        ax.set_title(label, fontsize=9, pad=4)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(families)), [f"held out: {f}" for f in families])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=7.5)
    top = title(
        fig,
        "Do the intervals know when the model is extrapolating?",
        "Filled = family held out of training; hollow = same family, in-distribution "
        f"(grouped split). {FEATURES} features; vertical line = nominal.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def pit_hist(hist):
    h = hist[(hist.model == "tabpfn") & (hist.kind == "grouped") & (hist.features == FEATURES)]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(
        h.bin_lo,
        h.density,
        width=0.05,
        align="edge",
        color=COLORS["tabpfn"],
        edgecolor=SURFACE,
        linewidth=0.8,
    )
    ax.axhline(1, color=INK_2, linewidth=0.8)
    ax.set_xlabel("PIT: predicted CDF at the true Tc")
    ax.set_ylabel("Density (1 = calibrated)")
    top = title(
        fig,
        "TabPFN-3.5 PIT histogram, grouped split",
        f"{FEATURES} features, 25 splits pooled. Flat = calibrated; U-shape = too narrow.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def p77_reliability(rel, brier):
    r = rel[(rel.model == "tabpfn") & (rel.kind == "grouped") & (rel.features == FEATURES)]
    b = brier[
        (brier.model == "tabpfn") & (brier.kind == "grouped") & (brier.features == FEATURES)
    ].iloc[0]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=0.8)
    ax.scatter(
        r.predicted,
        r.observed,
        s=np.clip(r.n / 40, 12, 160),
        color=COLORS["tabpfn"],
        edgecolors=SURFACE,
        linewidths=1,
        zorder=3,
    )
    ax.plot(r.predicted, r.observed, color=COLORS["tabpfn"], linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(Tc > 77 K), binned")
    ax.set_ylabel("Observed share with Tc > 77 K")
    ax.text(
        0.03,
        0.95,
        f"Brier {b.brier:.3f} (base rate only: {b.base_rate_brier:.3f})",
        fontsize=8,
        color=INK_2,
        va="top",
    )
    top = title(
        fig,
        "TabPFN-3.5: is P(Tc > 77 K) reliable?",
        "Grouped split, 25 splits pooled; marker area ~ rows per bin.",
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def main():
    plots.setup()
    curves = pd.read_csv(LC / "curves.csv")
    cal = pd.read_csv(UNC / "calibration.csv")
    figures = {
        "learning_curves": learning_curves(curves),
        "calibration_curve": calibration_curve(pd.read_csv(UNC / "curve.csv"), cal),
        "lfo_coverage": lfo_coverage(pd.read_csv(UNC / "lfo.csv")),
        "pit_hist": pit_hist(pd.read_csv(UNC / "pit_hist.csv")),
        "p77_reliability": p77_reliability(
            pd.read_csv(UNC / "p77_reliability.csv"), pd.read_csv(UNC / "p77_brier.csv")
        ),
    }
    for name, fig in figures.items():
        plots.save(fig, FIG / f"{name}.png")
        print(f"Wrote {FIG / name}.png")


if __name__ == "__main__":
    main()
