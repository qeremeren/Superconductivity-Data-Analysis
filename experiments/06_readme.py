"""README tables, rendered from committed results so no number in them is typed by hand.

    uv run python -m experiments.06_readme            # rewrite the blocks in README.md
    uv run python -m experiments.06_readme --check    # fail if README.md is out of date

Each block sits between `<!-- readme:NAME:start -->` and `<!-- readme:NAME:end -->` in
README.md and names the results file it comes from. tests/test_readme.py runs --check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import pandas as pd

from src import config

R = config.RESULTS_DIR
README = config.ROOT / "README.md"

MODEL = {
    "tabpfn": "TabPFN-3.5",
    "xgb_tuned": "XGBoost, tuned",
    "xgb_hamidieh": "XGBoost, published settings",
    "nn_lookup": "1-NN composition lookup",
    "linear": "Linear regression",
    "quantile_regression": "XGBoost quantile regression",
    "conformal": "XGBoost + split conformal",
}
ACQ = {
    "ei": "**TabPFN, expected improvement**",
    "greedy_tabpfn": "TabPFN, greedy (mean)",
    "greedy_xgb": "XGBoost, greedy (published settings)",
    "gp_ei": "Gaussian process, expected improvement",
    "random": "Random search",
}
SPLITS = ("random", "grouped", "grouped_no_oxygen")


def md(df: pd.DataFrame, align: str | None = None) -> str:
    cols = list(df.columns)
    align = align or "l" + "r" * (len(cols) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def source(*paths: str) -> str:
    return "<sub>Source: " + ", ".join(f"`results/{p}`" for p in paths) + "</sub>"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


# --- blocks -----------------------------------------------------------------------------


def benchmark() -> str:
    m = pd.read_csv(R / "02_benchmark/metrics.csv")
    m = m[(m.fold == "all splits") & (m.group == "all")]
    rows = [
        ("tabpfn", "composition"), ("tabpfn", "engineered"),
        ("xgb_tuned", "engineered"), ("xgb_tuned", "composition"),
        ("xgb_hamidieh", "engineered"), ("xgb_hamidieh", "composition"),
        ("nn_lookup", "composition"), ("linear", "engineered"),
    ]  # fmt: skip
    out = []
    for model, feats in rows:
        sel = m[(m.model == model) & (m.features == feats)].set_index("kind")
        name = MODEL[model] if model != "tabpfn" else f"**{MODEL[model]}**"
        out.append(
            [name, feats]
            + [f"{sel.loc[k, 'rmse']:.2f}" for k in SPLITS]
            + [f"{sel.loc['grouped', 'mae']:.2f}", f"{sel.loc['grouped', 'r2']:.3f}"]
        )
    df = pd.DataFrame(
        out,
        columns=[
            "Model", "Features", "RMSE random", "RMSE grouped", "RMSE grouped, no O",
            "MAE grouped", "R² grouped",
        ],
    )  # fmt: skip
    return md(df, "llrrrrr") + "\n\n" + source("02_benchmark/metrics.csv")


def replication() -> str:
    rep = json.loads((R / "02_benchmark/replication.json").read_text())
    out = []
    for model in ("xgb_hamidieh", "linear"):
        paper, ours = rep[model]["paper"], rep[model]["engineered"]
        out.append(
            [MODEL[model], f"{paper['rmse']:.1f}", f"{ours['rmse']:.2f}",
             f"{paper['r2']:.2f}", f"{ours['r2']:.3f}"]
        )  # fmt: skip
    df = pd.DataFrame(
        out, columns=["Model (engineered features)", "Paper RMSE", "Ours", "Paper R²", "Ours"]
    )
    return md(df) + "\n\n" + source("02_benchmark/replication.json")


def paired_benchmark() -> str:
    p = pd.DataFrame(json.loads((R / "02_benchmark/paired.json").read_text()))
    p = p[(p.a == "tabpfn") & (p.b == "xgb_tuned") & (p.metric == "rmse")]
    out = [
        [k, f, f"{r.mean_diff:+.2f}", f"{r.std_diff:.2f}", f"{r.a_better_splits}/{r.n_splits}"]
        for k in SPLITS
        for f in ("composition", "engineered")
        for r in p[(p.kind == k) & (p.features == f)].itertuples()
    ]
    df = pd.DataFrame(
        out,
        columns=["Split", "Features", "Mean RMSE difference (K)", "SD", "TabPFN better on"],
    )
    return md(df, "llrrr") + "\n\n" + source("02_benchmark/paired.json")


def subgroups() -> str:
    m = pd.read_csv(R / "02_benchmark/metrics.csv")
    m = m[(m.kind == "grouped") & (m.fold == "all splits")]
    groups = [
        "Tc < 10 K", "10-77 K", "Tc > 77 K", "family: cuprate", "family: iron-based",
        "family: other", "oxygen amount missing",
    ]  # fmt: skip
    cols = [("tabpfn", "composition"), ("xgb_tuned", "composition"), ("xgb_hamidieh", "engineered")]
    out = []
    for g in groups:
        sel = m[m.group == g]
        rmse = [sel[(sel.model == mo) & (sel.features == f)].rmse.iloc[0] for mo, f in cols]
        n = sel.rows_per_split.iloc[0]
        out.append([g, f"{n:.0f}"] + [f"{v:.2f}" for v in rmse])
    df = pd.DataFrame(
        out,
        columns=[
            "Test rows (grouped split)", "Rows per split", "TabPFN-3.5, composition",
            "XGBoost tuned, composition", "XGBoost published, engineered",
        ],
    )  # fmt: skip
    return md(df) + "\n\n" + source("02_benchmark/metrics.csv")


def calibration() -> str:
    c = pd.read_csv(R / "03_uncertainty/calibration.csv")
    c = c[(c.kind == "grouped") & (c.features == "composition") & (c.fold == "all splits")]
    c = c[c.group == "all"].set_index("model")
    b = pd.read_csv(R / "03_uncertainty/p77_brier.csv")
    b = b[(b.kind == "grouped") & (b.features == "composition")].set_index("model")
    out = []
    for model in ("tabpfn", "conformal", "quantile_regression"):
        r = c.loc[model]
        name = MODEL[model] if model != "tabpfn" else f"**{MODEL[model]}**"
        out.append(
            [name]
            + [pct(r[f"coverage_{lv}"]) for lv in ("50%", "80%", "90%", "95%")]
            + [f"{r['width_95%']:.1f} K", f"{r.crps20:.2f} K", f"{b.loc[model, 'brier']:.3f}"]
        )
    df = pd.DataFrame(
        out,
        columns=[
            "Model", "50% interval coverage", "80%", "90%", "95%", "95% width",
            "CRPS", "Brier, Tc > 77 K",
        ],
    )  # fmt: skip
    base = b.base_rate_brier.iloc[0]
    return (
        md(df) + f"\n\nCRPS on the same 20 quantile levels for every model. Brier score of always "
        f"predicting the base rate: {base:.3f}.\n\n"
        + source("03_uncertainty/calibration.csv", "03_uncertainty/p77_brier.csv")
    )


def learning_curves() -> str:
    c = pd.read_csv(R / "03_learning_curves/curves.csv")
    sizes = ["100", "300", "1000", "3000", "10000", "full"]
    out = []
    for feats in ("composition", "engineered"):
        for model in ("tabpfn", "xgb_hamidieh", "xgb_tuned"):
            sel = c[(c.model == model) & (c.features == feats)].set_index("n_label")
            name = MODEL[model] if model != "tabpfn" else f"**{MODEL[model]}**"
            out.append([feats, name] + [f"{sel.loc[s, 'rmse']:.2f}" for s in sizes])
    df = pd.DataFrame(
        out, columns=["Features", "Model", "100", "300", "1k", "3k", "10k", "full (~14.2k)"]
    )
    return (
        md(df, "llrrrrrr")
        + "\n\nRMSE (K) on the fixed test sets of grouped splits 0-4, nested training subsets.\n\n"
        + source("03_learning_curves/curves.csv")
    )


def leave_family_out() -> str:
    lfo = pd.read_csv(R / "03_uncertainty/lfo.csv")
    lfo = lfo[lfo.features == "composition"]
    out = []
    for fam in ("cuprate", "iron-based", "other"):
        row = [fam]
        for model in ("tabpfn", "quantile_regression", "conformal"):
            r = lfo[(lfo.model == model) & (lfo.held_out_family == fam)].iloc[0]
            row.append(f"{pct(r['held_out_coverage_95%'])} ({r['held_out_width_95%']:.0f} K)")
        t = lfo[(lfo.model == "tabpfn") & (lfo.held_out_family == fam)].iloc[0]
        cov, width = t["in_distribution_coverage_95%"], t["in_distribution_width_95%"]
        row.append(f"{pct(cov)} ({width:.0f} K)")
        row.append(f"{t.held_out_rmse:.1f} K")
        out.append(row)
    df = pd.DataFrame(
        out,
        columns=[
            "Held-out family", "TabPFN 95% coverage (width)", "XGBoost QR", "XGBoost conformal",
            "TabPFN in-distribution", "TabPFN RMSE held out",
        ],
    )  # fmt: skip
    return md(df) + "\n\n" + source("03_uncertainty/lfo.csv")


def discovery() -> str:
    s = pd.read_csv(R / "04_discovery/summary.csv")
    s = s[s.phase == "main"].set_index(["scenario", "acquisition"])
    out = []
    for acq in ("ei", "greedy_tabpfn", "greedy_xgb", "gp_ei", "random"):
        row = [ACQ[acq]]
        for scn in ("main", "hard"):
            r = s.loc[(scn, acq)]
            row.append(f"{r.found_200:.1f} ({r.found_200_lo:.1f} to {r.found_200_hi:.1f})")
            row.append(f"{r.first_hit_median:.0f}")
            row.append(f"{r.runs_without_hit:.0f}/{r.seeds:.0f}")
        out.append(row)
    main, hard = s.loc[("main", "ei")], s.loc[("hard", "ei")]
    df = pd.DataFrame(
        out,
        columns=[
            "Method", "Main: targets found", "first hit", "no hit",
            "Hard: targets found", "first hit", "no hit",
        ],
    )  # fmt: skip
    return (
        md(df) + f"\n\nTargets found after 200 experiments, mean over 10 seeds (95% t-interval); "
        f'"first hit" is the median number of experiments to the first target. The main pool '
        f"has {main.targets_in_pool:.0f} targets among 15,147 materials, the hard pool "
        f"{hard.targets_in_pool:.0f} among 7,637. Random search expects "
        f"{main.random_200:.2f} and {hard.random_200:.2f} targets in 200 picks (exact "
        f"hypergeometric mean).\n\n" + source("04_discovery/summary.csv")
    )


def discovery_paired() -> str:
    p = pd.read_csv(R / "04_discovery/paired.csv")
    p = p[(p.phase == "main") & (p.a == "ei")]
    out = []
    short = {
        "greedy_tabpfn": "TabPFN greedy (same model, same first fit)",
        "greedy_xgb": "XGBoost greedy",
        "gp_ei": "GP expected improvement",
        "random": "random search",
    }
    for b, label in short.items():
        row = [f"EI minus {label}"]
        for scn in ("main", "hard"):
            for n in (100, 200):
                r = p[(p.scenario == scn) & (p.b == b) & (p.experiments == n)].iloc[0]
                row.append(
                    f"{r.mean_diff:+.1f} ({r.diff_lo:+.1f} to {r.diff_hi:+.1f}); "
                    f"{r.a_better}/{r.seeds}"
                )
        out.append(row)
    df = pd.DataFrame(
        out,
        columns=["TabPFN EI vs", "Main, 100 exp.", "Main, 200 exp.", "Hard, 100", "Hard, 200"],
    )
    return (
        md(df)
        + "\n\nMean difference in targets found (95% interval); seeds on which EI found more.\n\n"
        + source("04_discovery/paired.csv")
    )


def demo() -> str:
    s = pd.read_csv(R / "06_demo/showcase.csv")
    out = []
    for _, r in s.iterrows():
        n = int(r.in_data_rows)
        recorded = f"{r.recorded_tc_median:.1f} ({n} rows)" if n else "not in data"
        p = r.p_above_77K
        p_txt = "< 0.5%" if p <= 0.005 + 1e-9 else ("> 99.5%" if p >= 0.995 - 1e-9 else pct(p))
        out.append(
            [f"`{r.formula}`", r.family, recorded, f"{r['median']:.1f}",
             f"{r['q0.1']:.1f} to {r['q0.9']:.1f}", f"{r['q0.025']:.1f} to {r['q0.975']:.1f}",
             p_txt, f"{r.nearest_l1:.3f}"]
        )  # fmt: skip
    df = pd.DataFrame(
        out,
        columns=[
            "Formula", "Family", "Recorded Tc, K (median)", "Predicted median", "80% interval",
            "95% interval", "P(Tc > 77 K)", "Nearest training material (L1)",
        ],
    )  # fmt: skip
    return md(df, "llrrrrrr") + "\n\n" + source("06_demo/showcase.csv")


BLOCKS = {
    "benchmark": benchmark,
    "replication": replication,
    "paired-benchmark": paired_benchmark,
    "subgroups": subgroups,
    "calibration": calibration,
    "learning-curves": learning_curves,
    "leave-family-out": leave_family_out,
    "discovery": discovery,
    "discovery-paired": discovery_paired,
    "demo": demo,
}


def render(text: str) -> str:
    for name, fn in BLOCKS.items():
        start, end = f"<!-- readme:{name}:start -->", f"<!-- readme:{name}:end -->"
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
        if not pattern.search(text):
            raise KeyError(f"README.md has no block {name!r}")
        block = f"{start}\n{fn()}\n{end}"
        text = pattern.sub(lambda _m, block=block: block, text)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    current = README.read_text()
    updated = render(current)
    if args.check:
        if updated != current:
            print("README.md tables are out of date: run `make readme`", file=sys.stderr)
            return 1
        print("README.md tables match results/")
        return 0
    README.write_text(updated)
    print(f"updated {len(BLOCKS)} blocks in {README}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
