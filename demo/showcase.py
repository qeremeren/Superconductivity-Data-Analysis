"""The demo's showcase: 16 well-known superconductors from the data (each held out of
training for its own prediction) and 4 materials that are not in the data.

    uv run python -m demo.showcase           # table + figure from the committed cache (no API)
    uv run python -m demo.showcase --live    # author: fill missing predictions, 1 request each

Writes results/06_demo/showcase.csv and results/06_demo/figures/showcase.png.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from demo import predict
from src import config, data, plots
from src.plots import INK, INK_2, MUTED

# Input formulas only; every Tc shown comes from the data or from the model.
IN_DATA = (
    "Hg", "Pb", "Nb", "NbN", "Nb3Sn", "MgB2", "K3C60",
    "FeSe", "LaFeAsO0.9F0.1", "Ba0.6K0.4Fe2As2", "SmFeAsO0.85F0.15",
    "La1.85Sr0.15CuO4", "YBa2Cu3O7", "Bi2Sr2Ca2Cu3O10", "Tl2Ba2Ca2Cu3O10", "HgBa2Ca2Cu3O8",
)  # fmt: skip
NOT_IN_DATA = ("CsV3Sb5", "Nd0.8Sr0.2NiO2", "La3Ni2O7", "LaH10")
FORMULAS = IN_DATA + NOT_IN_DATA


def figure(table: pd.DataFrame) -> None:
    plots.setup()
    rows = table.iloc[::-1].reset_index(drop=True)  # first formula at the top
    n_new = len(NOT_IN_DATA)
    y = np.arange(len(rows), dtype=float)
    y[n_new:] += 0.8  # gap between the two groups
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for yi, (_, r) in zip(y, rows.iterrows(), strict=True):
        c = plots.FAMILY_COLORS[r.family]
        ax.plot([r["q0.025"], r["q0.975"]], [yi, yi], color=c, lw=2, alpha=0.35)
        ax.plot([r["q0.1"], r["q0.9"]], [yi, yi], color=c, lw=6, alpha=0.75)
        ax.plot(r["median"], yi, "o", color="white", ms=4, mec=c, mew=1.2, zorder=3)
        if r.in_data_rows:
            ax.plot(r.recorded_tc_median, yi, "D", color=INK, ms=4.5, zorder=4)
    ax.set_yticks(y, rows.formula)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.grid(axis="y", visible=False)
    ax.axvline(77, color=MUTED, lw=0.8, ls=":")
    ax.text(77, y[-1] + 1.0, " 77 K", color=INK_2, va="center", fontsize=8)
    sep = (y[n_new - 1] + y[n_new]) / 2
    ax.axhline(sep, color=MUTED, lw=0.6)
    ax.text(
        ax.get_xlim()[1], sep - 0.45, "not in the data: no recorded Tc ", ha="right",
        va="center", color=INK_2, fontsize=8,
    )  # fmt: skip
    ax.set_xlabel("Tc (K)")
    ax.set_xlim(left=min(-5, rows["q0.025"].min() - 5))
    ax.set_ylim(-0.8, y[-1] + 1.6)
    ax.set_title("TabPFN-3.5 predictive distributions, each material held out of training")
    handles = [
        plt.Line2D([], [], color=MUTED, lw=6, alpha=0.75, label="80% interval"),
        plt.Line2D([], [], color=MUTED, lw=2, alpha=0.35, label="95% interval"),
        plt.Line2D([], [], ls="", marker="o", color="white", mec=MUTED, label="median"),
        plt.Line2D([], [], ls="", marker="D", color=INK, ms=4.5, label="recorded Tc (median)"),
    ] + [
        plt.Line2D([], [], color=plots.FAMILY_COLORS[f], lw=6, label=plots.FAMILY_LABELS[f])
        for f in (data.CUPRATE, data.IRON_BASED, data.OTHER)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7.5, ncol=2)
    plots.save(fig, predict.OUT / "figures" / "showcase.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    results = predict.run(list(FORMULAS), config.live_requested(args.live))
    table = pd.DataFrame(results)
    assert (table.in_data_rows[: len(IN_DATA)] > 0).all(), "a showcase material left the data"
    assert (table.in_data_rows[len(IN_DATA) :] == 0).all(), "a 'not in data' material is in it"
    table.round(4).to_csv(predict.OUT / "showcase.csv", index=False)
    figure(table)
    print(f"wrote {predict.OUT / 'showcase.csv'} and figures/showcase.png ({len(table)} formulas)")


if __name__ == "__main__":
    main()
