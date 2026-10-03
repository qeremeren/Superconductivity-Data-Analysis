"""Phase 5: do the predictive intervals widen with distance from the training data?

Composition features, from the committed Phase 2/3 per-row outputs (no API calls, no fitting):
TabPFN-3.5 and the two XGBoost uncertainty baselines (quantile regression, conformal), on the
grouped split (25 repeats) and the leave-family-out folds. Distance = L1 over element fractions
to the nearest training material (benchmark.nearest_training_all). Writes results/05_why/:
  interval_vs_distance.csv  per split kind, held-out family (LFO) or "all" (grouped), test
                            family (grouped: also per family), model and distance bin: rows,
                            80% / 95% coverage and mean width, RMSE
  interval_vs_neighbour_spread.csv  the same by the SD of Tc among the 10 nearest training
                            materials (benchmark.neighbour_tc_spread)
  width_drivers.csv         Spearman correlation of the 80% width with distance, with that
                            neighbour Tc SD and with |error|, per split, averaged over the
                            grouped splits; per LFO fold
  lfo_neighbours.csv        per held-out family: how far the nearest training material is,
                            which family and Tc it has, how TabPFN's prediction relates to it,
                            and the most common chemical systems of those neighbours
  figures/interval_width_drivers.png

Run from the repo root: `uv run python -m experiments.05_intervals`.
"""

from __future__ import annotations

import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import benchmark, config, data, plots, splits
from src.plots import FAMILY_COLORS, FAMILY_LABELS, INK, INK_2, SURFACE

OUT = config.RESULTS_DIR / "05_why"
SOURCES = {
    "tabpfn": benchmark.OUT / "tabpfn",
    "quantile_regression": config.RESULTS_DIR / "03_uncertainty" / "xgb" / "quantile_regression",
    "conformal": config.RESULTS_DIR / "03_uncertainty" / "xgb" / "conformal",
}
KINDS = ("grouped", "leave_family_out")
COLS = ["row", "y", "mean", "q0.025", "q0.1", "q0.9", "q0.975"]


def load(model: str, kind: str) -> pd.DataFrame:
    paths = sorted((SOURCES[model] / kind / "composition").glob("split_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {model} outputs for {kind}")
    d = pd.concat(
        [pd.read_parquet(p, columns=COLS).assign(split=int(p.stem[-2:])) for p in paths],
        ignore_index=True,
    )
    return d.assign(
        width80=d["q0.9"] - d["q0.1"],
        width95=d["q0.975"] - d["q0.025"],
        cover80=(d.y >= d["q0.1"]) & (d.y <= d["q0.9"]),
        cover95=(d.y >= d["q0.025"]) & (d.y <= d["q0.975"]),
        abs_err=np.abs(d["mean"] - d.y),
    )


def held_out(kind: str, split: pd.Series) -> pd.Series:
    if kind == "grouped":
        return pd.Series("all", index=split.index)
    return split.map(dict(enumerate(splits.FAMILY_ORDER)))


def system(key: str) -> str:
    return "-".join(sorted(part.split(":")[0] for part in key.split()))


SPREAD_BINS = (0.0, 2.0, 5.0, 10.0, 20.0, 30.0, np.inf)
SPREAD_LABELS = ("< 2", "2-5", "5-10", "10-20", "20-30", ">= 30")


def summarise(g: pd.DataFrame) -> dict:
    return {
        "rows": len(g),
        "coverage_80": g.cover80.mean(),
        "coverage_95": g.cover95.mean(),
        "width_80": g.width80.mean(),
        "width_95": g.width95.mean(),
        "rmse": float(np.sqrt(((g["mean"] - g.y) ** 2).mean())),
    }


def rank(a, b, constant) -> float:
    return np.nan if constant else stats.spearmanr(a, b).statistic


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    um = data.load_unique_m()
    fam = data.family(um).to_numpy()
    by_dist, by_spread, ranks, frames = [], [], [], {}
    for kind in KINDS:
        nn = benchmark.nearest_training_all(kind, um).merge(
            benchmark.neighbour_tc_spread_all(kind, um), on=["split", "row"], validate="1:1"
        )
        for model in SOURCES:
            d = load(model, kind).merge(nn, on=["split", "row"], validate="one_to_one")
            d = d.assign(
                held_out=held_out(kind, d.split),
                family=fam[d.row],
                distance=benchmark.nn_bin(d.nn_distance),
                spread=pd.cut(d.knn10_tc_sd, SPREAD_BINS, right=False, labels=SPREAD_LABELS),
            )
            frames[kind, model] = d
            groups = [("all", d)] + (
                [(f, g) for f, g in d.groupby("family")] if kind == "grouped" else []
            )
            for f, sub in groups:
                for (h, b), g in sub.groupby(["held_out", "distance"], observed=True):
                    by_dist.append(
                        {
                            "split_kind": kind,
                            "held_out": h,
                            "family": f,
                            "model": model,
                            "distance": b,
                        }
                        | summarise(g)
                    )
            for (h, b), g in d.groupby(["held_out", "spread"], observed=True):
                by_spread.append(
                    {"split_kind": kind, "held_out": h, "model": model, "knn10_tc_sd": b}
                    | summarise(g)
                )
            for (h, s), g in d.groupby(["held_out", "split"]):
                const = g.width80.std() <= 1e-6 * max(1.0, g.width80.abs().mean())
                ranks.append(
                    {
                        "split_kind": kind,
                        "held_out": h,
                        "model": model,
                        "split": s,
                        "spearman_width_distance": rank(g.width80, g.nn_distance, const),
                        "spearman_width_knn10_tc_sd": rank(g.width80, g.knn10_tc_sd, const),
                        "spearman_width_abs_error": rank(g.width80, g.abs_err, const),
                    }
                )
    by_dist, by_spread = pd.DataFrame(by_dist), pd.DataFrame(by_spread)
    cols = ["spearman_width_distance", "spearman_width_knn10_tc_sd", "spearman_width_abs_error"]
    ranks = pd.DataFrame(ranks).groupby(["split_kind", "held_out", "model"])[cols].mean()
    ranks = ranks.reset_index()
    by_dist.to_csv(OUT / "interval_vs_distance.csv", index=False, float_format="%.4f")
    by_spread.to_csv(OUT / "interval_vs_neighbour_spread.csv", index=False, float_format="%.4f")
    ranks.to_csv(OUT / "width_drivers.csv", index=False, float_format="%.4f")

    rows = []
    for h, g in frames["leave_family_out", "tabpfn"].groupby("held_out"):
        top = collections.Counter(map(system, g.nn_key)).most_common(5)
        rows.append(
            {
                "held_out": h,
                "rows": len(g),
                "nn_distance_median": g.nn_distance.median(),
                "share_nn_within_0.05": (g.nn_distance < 0.05).mean(),
                **{f"share_nn_family_{f}": (g.nn_family == f).mean() for f in splits.FAMILY_ORDER},
                "tc_median": g.y.median(),
                "nn_tc_median": g.nn_tc.median(),
                "knn10_tc_mean_median": g.knn10_tc_mean.median(),
                "knn10_tc_sd_median": g.knn10_tc_sd.median(),
                "tabpfn_pred_median": g["mean"].median(),
                "spearman_tabpfn_pred_vs_knn10_tc_mean": stats.spearmanr(
                    g["mean"], g.knn10_tc_mean
                ).statistic,
                "tabpfn_width80_median": g.width80.median(),
                "tabpfn_coverage_80": g.cover80.mean(),
                "top_nn_systems": "; ".join(f"{s} ({c / len(g):.0%})" for s, c in top),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "lfo_neighbours.csv", index=False, float_format="%.4f")
    plots.setup()
    plots.save(figure(by_spread, by_dist), OUT / "figures" / "interval_width_drivers.png")
    t = by_dist[(by_dist.model == "tabpfn") & (by_dist.split_kind == "grouped")]
    print(t.round(3).to_string(index=False))
    print(by_spread[by_spread.model == "tabpfn"].round(3).to_string(index=False))
    print(ranks.round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(3).T.to_string())


def figure(by_spread: pd.DataFrame, by_dist: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    sp = by_spread[(by_spread.model == "tabpfn") & (by_spread.rows >= 30)]
    series = [("grouped", "all", INK_2, "Grouped split")] + [
        ("leave_family_out", f, FAMILY_COLORS[f], f"{FAMILY_LABELS[f]} held out")
        for f in splits.FAMILY_ORDER
    ]
    xs = {lab: i for i, lab in enumerate(SPREAD_LABELS)}
    for kind, h, color, label in series:
        d = sp[(sp.split_kind == kind) & (sp.held_out == h)]
        axes[0].plot(
            [xs[v] for v in d.knn10_tc_sd],
            d.width_80,
            color=color,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            label=label,
        )
    axes[0].set_xticks(range(len(SPREAD_LABELS)), SPREAD_LABELS, fontsize=8)
    axes[0].set_xlabel("SD of Tc among the 10 nearest training materials (K)")
    axes[0].set_ylabel("Mean width of TabPFN's 80% interval (K)")
    axes[0].set_title("Width vs the Tc spread of the 10 nearest materials", fontsize=9, pad=4)
    axes[0].legend(loc="upper left", fontsize=7.5)
    dd = by_dist[
        (by_dist.model == "tabpfn")
        & (by_dist.split_kind == "grouped")
        & (by_dist.family != "all")
        & (by_dist.rows >= 30)
    ]
    xd = {lab: i for i, lab in enumerate(benchmark.NN_LABELS)}
    for f in splits.FAMILY_ORDER:
        d = dd[dd.family == f]
        axes[1].plot(
            [xd[v] for v in d.distance],
            d.width_80,
            color=FAMILY_COLORS[f],
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            label=FAMILY_LABELS[f],
        )
    axes[1].set_xticks(range(len(benchmark.NN_LABELS)), benchmark.NN_LABELS, fontsize=8)
    axes[1].set_xlabel("Distance to the nearest training material (L1)")
    axes[1].set_title("Grouped split: width by distance, per family", fontsize=9, pad=4)
    axes[1].legend(loc="upper left", fontsize=7.5)
    h = fig.get_figheight()
    fig.suptitle(
        "What sets the width of TabPFN's intervals?",
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
        "Composition features; bins with fewer than 30 rows omitted.",
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / h))
    return fig


if __name__ == "__main__":
    main()
