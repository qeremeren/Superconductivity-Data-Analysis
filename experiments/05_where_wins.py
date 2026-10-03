"""Phase 5: where TabPFN-3.5 beats tuned XGBoost on the primary benchmark, and where it doesn't.

Grouped split (25 repeats), both feature sets, from the committed Phase 2 predictions (no API
calls, no fitting). Every test row gets the L1 distance to its nearest training material
(benchmark.nearest_training_material). Writes results/05_why/:
  where_by_distance.csv  per feature set and distance bin: rows, RMSE of both models (pooled
                         over splits), per-split paired RMSE difference (mean, 95% t-interval,
                         splits TabPFN better), per-row win rate, and the bin's share of the
                         total MSE advantage
  where_by_subgroup.csv  the same by family, Tc band, missing-oxygen rows, and family x distance
  worst_materials.csv    the 25 materials with the largest TabPFN error (composition features,
                         mean over the splits that test them), with both models' predictions,
                         the Tc spread among the material's own duplicate rows, distance
  worst_summary.json     how the worst 1% of materials differ from all materials
Per-row distances are cached (gitignored) by benchmark.nearest_training_all.

Run from the repo root: `uv run python -m experiments.05_where_wins`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src import benchmark, config, data

OUT = config.RESULTS_DIR / "05_why"
FEATURES = ("composition", "engineered")


def predictions(features: str) -> pd.DataFrame:
    tab = pd.concat(
        [
            pd.read_parquet(p, columns=["row", "y", "mean"]).assign(split=int(p.stem[-2:]))
            for p in sorted(
                (benchmark.OUT / "tabpfn" / "grouped" / features).glob("split_*.parquet")
            )
        ],
        ignore_index=True,
    )
    xgb = pd.read_parquet(benchmark.predictions_path("xgb_tuned", "grouped", features))
    m = tab.merge(xgb, on=["split", "row"], how="inner", validate="one_to_one")
    if len(m) != len(tab) or len(m) != len(xgb):
        raise ValueError(f"{features}: TabPFN and XGBoost test rows differ")
    return m.rename(columns={"mean": "tabpfn", "pred": "xgb"})


def ci(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def compare(d: pd.DataFrame, total_adv: float, n_total: int) -> dict:
    se_t, se_x = (d.tabpfn - d.y) ** 2, (d.xgb - d.y) ** 2
    per_split = d.assign(se_t=se_t, se_x=se_x).groupby("split")[["se_t", "se_x"]].mean()
    diff = np.sqrt(per_split.se_t) - np.sqrt(per_split.se_x)
    m, lo, hi = ci(diff) if len(diff) > 1 else (np.nan, np.nan, np.nan)
    return {
        "rows": len(d),
        "share_of_rows": len(d) / n_total,
        "rmse_tabpfn": float(np.sqrt(se_t.mean())),
        "rmse_xgb": float(np.sqrt(se_x.mean())),
        "rmse_diff_per_split_mean": m,
        "rmse_diff_lo": lo,
        "rmse_diff_hi": hi,
        "splits": len(diff),
        "splits_tabpfn_better": int((diff < 0).sum()),
        "row_win_rate_tabpfn": float((np.abs(d.tabpfn - d.y) < np.abs(d.xgb - d.y)).mean()),
        "share_of_mse_advantage": float((se_x.sum() - se_t.sum()) / n_total / total_adv),
    }


def by_groups(d: pd.DataFrame, features: str, groupings: dict) -> list[dict]:
    n = len(d)
    total_adv = float(((d.xgb - d.y) ** 2).mean() - ((d.tabpfn - d.y) ** 2).mean())
    rows = []
    for name, col in groupings.items():
        for value, g in d.groupby(col, observed=True):
            label = " / ".join(map(str, value)) if isinstance(value, tuple) else str(value)
            rows.append(
                {"features": features, "grouping": name, "group": label} | compare(g, total_adv, n)
            )
    return rows


def worst(d: pd.DataFrame, um: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    key = data.composition_key(um)
    tc = um[data.TARGET]
    mat = pd.DataFrame({"key": key, "tc": tc}).groupby("key").tc
    spread = (mat.max() - mat.min()).rename("tc_range_within_material")
    d = d.assign(
        key=key.to_numpy()[d.row],
        err_t=np.abs(d.tabpfn - d.y),
        err_x=np.abs(d.xgb - d.y),
    )
    per = d.groupby("key").agg(
        formula=("row", lambda r: um[data.FORMULA].iloc[r.iloc[0]]),
        family=("family", "first"),
        n_rows=("row", "nunique"),
        tc_median=("y", "median"),
        tabpfn_pred=("tabpfn", "mean"),
        xgb_pred=("xgb", "mean"),
        tabpfn_abs_err=("err_t", "mean"),
        xgb_abs_err=("err_x", "mean"),
        nn_distance=("nn_distance", "mean"),
        nn_tc=("nn_tc", "mean"),
        oxygen_missing=("oxygen_missing", "max"),
    )
    suspect = (
        pd.Series(data.suspect_reasons(um).to_numpy(), index=key.to_numpy())
        .groupby(level=0)
        .agg(lambda r: "; ".join(sorted({x for x in r if x})))
        .rename("suspect_phase1")
    )
    per = per.join(spread).join(suspect)
    per = per.sort_values("tabpfn_abs_err", ascending=False)
    top = per.head(max(1, len(per) // 100))

    def describe(x):
        return {
            "materials": len(x),
            "median_tc_range_within_material_K": float(x.tc_range_within_material.median()),
            "share_with_tc_range_over_20K": float((x.tc_range_within_material > 20).mean()),
            "share_duplicated": float((x.n_rows > 1).mean()),
            "share_oxygen_missing": float(x.oxygen_missing.mean()),
            "median_nn_distance": float(x.nn_distance.median()),
            "family_shares": {
                k: float(v) for k, v in x.family.value_counts(normalize=True).items()
            },
            "share_xgb_also_off_by_30K": float((x.xgb_abs_err > 30).mean()),
            "share_flagged_suspect_in_phase1": float((x.suspect_phase1 != "").mean()),
        }

    summary = {
        "worst_1pct_by_tabpfn_error": describe(top),
        "worst_25_by_tabpfn_error": describe(per.head(25)),
        "all_materials": describe(per),
    }
    return per.head(25).reset_index(), summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    um = data.load_unique_m()
    inputs = benchmark.load_inputs()
    nn = benchmark.nearest_training_all("grouped", um)
    rows, worst_summary = [], {}
    for features in FEATURES:
        d = predictions(features).merge(nn, on=["split", "row"], validate="one_to_one")
        if not np.allclose(d.y, inputs.y[d.row]):
            raise ValueError("Tc in the TabPFN cache does not match the data")
        d = d.assign(
            distance=benchmark.nn_bin(d.nn_distance),
            family=inputs.family[d.row],
            oxygen_missing=inputs.oxygen_missing[d.row],
            tc_band=pd.cut(
                d.y, [-np.inf, 10, 77, np.inf], right=False, labels=["< 10 K", "10-77 K", ">= 77 K"]
            ),
        )
        rows += by_groups(
            d,
            features,
            {
                "all": lambda _: "all",
                "distance": "distance",
                "family": "family",
                "tc_band": "tc_band",
                "oxygen_missing": "oxygen_missing",
                "family x distance": ["family", "distance"],
            },
        )
        if features == "composition":
            table, worst_summary = worst(d, um)
            table.to_csv(OUT / "worst_materials.csv", index=False, float_format="%.4g")
    res = pd.DataFrame(rows)
    res[res.grouping == "distance"].to_csv(
        OUT / "where_by_distance.csv", index=False, float_format="%.4f"
    )
    res[res.grouping != "distance"].to_csv(
        OUT / "where_by_subgroup.csv", index=False, float_format="%.4f"
    )
    (OUT / "worst_summary.json").write_text(json.dumps(worst_summary, indent=1) + "\n")
    print(res[res.grouping.isin(["all", "distance"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
