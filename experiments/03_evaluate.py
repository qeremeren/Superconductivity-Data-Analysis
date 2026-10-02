"""Phase 3 evaluation from committed files only (no API calls, no model fitting).

Learning curves -> results/03_learning_curves/
  per_job.csv        RMSE/MAE (and CRPS for TabPFN) per model, features, split, size
  curves.csv         mean and SD over the 5 splits per model, features, size
  paired.csv         TabPFN minus each XGBoost per features and size (mean, SD, wins)
  machines.json      which machine produced each local (XGBoost) result
Uncertainty -> results/03_uncertainty/
  per_split.parquet  coverage/width at 50/80/90/95%, CRPS (100- and 20-level), per split/group
  calibration.csv    the above averaged over splits
  curve.csv          nominal vs empirical central-interval coverage (from PIT, levels <= 0.9)
  pit_hist.csv       PIT histograms (20 bins)
  p77_reliability.csv  predicted P(Tc > 77 K) vs observed frequency, and Brier scores
  lfo.csv            leave-family-out vs the same family in-distribution (grouped split),
                     with the per-row rank correlation between interval width and |error|
XGBoost files appear after the off-Mac runs; until then they are skipped (or, with
--require-all, a missing file is an error).

Run from the repo root: `uv run python -m experiments.03_evaluate [--require-all]`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import benchmark, data, metrics, splits

RES = benchmark.config.RESULTS_DIR
LC = RES / "03_learning_curves"
UNC = RES / "03_uncertainty"
REPEATED = ("random", "grouped", "grouped_no_oxygen")
XGB_KINDS = ("grouped", "leave_family_out")
XGB_METHODS = ("quantile_regression", "conformal")
CURVE_LEVELS = np.round(np.arange(0.1, 0.91, 0.1), 2)


class Missing(RuntimeError):
    pass


def need(path: Path, require_all: bool) -> bool:
    if path.is_file():
        return True
    if require_all:
        raise Missing(f"{path} is missing")
    return False


# --- learning curves -----------------------------------------------------------------


def lc_jobs(inputs, require_all):
    rows = []
    masks = splits.load("grouped")
    full_n = {s: int((~masks[s]).sum()) for s in benchmark.LC_SPLITS}
    for fs in data.FEATURE_SETS:
        for s in benchmark.LC_SPLITS:
            y_rows = np.flatnonzero(masks[s])
            for n in (*benchmark.LC_SIZES, full_n[s]):
                full = n == full_n[s]
                stem = f"split_{s:02d}_n{n:05d}"
                tab = (
                    RES / "02_benchmark/tabpfn/grouped" / fs / f"split_{s:02d}.parquet"
                    if full
                    else LC / "tabpfn/lc" / fs / f"{stem}.parquet"
                )
                if need(tab, require_all):
                    t = pd.read_parquet(tab)
                    m = metrics.point(t.y, t["mean"])
                    rows.append(
                        {
                            "model": "tabpfn",
                            "features": fs,
                            "split": s,
                            **m,
                            "n_train": n,
                            "crps": float(t.crps.mean()),
                        }
                    )
                for model in ("xgb_hamidieh", "xgb_tuned"):
                    if full:
                        p = pd.read_parquet(benchmark.predictions_path(model, "grouped", fs))
                        p = p[p.split == s]
                    else:
                        path = LC / "xgb/predictions" / model / fs / f"{stem}.parquet"
                        record = LC / "xgb/records" / model / fs / f"{stem}.json"
                        if not (need(path, require_all) and need(record, require_all)):
                            continue
                        p = pd.read_parquet(path)
                    assert (p.row.to_numpy() == y_rows).all()
                    m = metrics.point(inputs.y[y_rows], p.pred)
                    rows.append({"model": model, "features": fs, "split": s, **m, "n_train": n})
    df = pd.DataFrame(rows)
    # metrics.point's own "n" is the number of test rows scored; the size is n_train.
    df["n_label"] = np.where(df.n_train > max(benchmark.LC_SIZES), "full", df.n_train.astype(str))
    return df


def lc_tables(per_job):
    g = per_job.groupby(["model", "features", "n_label"], sort=False)
    curves = g.agg(
        n_train=("n_train", "mean"),
        splits=("split", "nunique"),
        rmse=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        mae=("mae", "mean"),
        mae_sd=("mae", "std"),
        crps=("crps", "mean"),
        crps_sd=("crps", "std"),
    ).reset_index()
    rmse = per_job.set_index(["model", "features", "n_label", "split"]).rmse.sort_index()
    paired = []
    for (fs, n_label), _ in per_job.groupby(["features", "n_label"]):
        try:
            tab = rmse.loc[("tabpfn", fs, n_label)]
        except KeyError:
            continue
        for other in ("xgb_tuned", "xgb_hamidieh"):
            if (other, fs, n_label) in rmse.index.droplevel("split"):
                paired.append(
                    {
                        "features": fs,
                        "n_label": n_label,
                        "b": other,
                        **metrics.paired(tab, rmse.loc[(other, fs, n_label)]),
                    }
                )
    return curves, pd.DataFrame(paired)


def lc_machines():
    hosts = {}
    for record in (LC / "xgb/records").glob("*/*/*.json"):
        host = json.loads(record.read_text())["machine"]["host"]
        key = record.parts[-3]
        hosts.setdefault(key, {}).setdefault(host, 0)
        hosts[key][host] += 1
    for record in (UNC / "xgb").glob("*/*/*/*.json"):
        host = json.loads(record.read_text())["machine"]["host"]
        key = f"uncertainty/{record.parts[-4]}"
        hosts.setdefault(key, {}).setdefault(host, 0)
        hosts[key][host] += 1
    return hosts


# --- uncertainty -----------------------------------------------------------------------


def load_distributions(require_all):
    """(model, kind, features, split) -> per-row summary with y, quantiles, pit, crps."""
    out = {}
    crps20 = pd.read_parquet(UNC / "tabpfn_crps20.parquet")
    crps20 = crps20.set_index(["kind", "features", "split", "row"]).crps20.sort_index()
    for kind in (*REPEATED, "leave_family_out"):
        for fs in data.FEATURE_SETS:
            for path in sorted((RES / "02_benchmark/tabpfn" / kind / fs).glob("split_*.parquet")):
                s = int(path.stem.split("_")[1])
                df = pd.read_parquet(path)
                if kind in XGB_KINDS:
                    df["crps20"] = crps20.loc[(kind, fs, s)].reindex(df.row).to_numpy()
                out["tabpfn", kind, fs, s] = df
    for method in XGB_METHODS:
        for kind in XGB_KINDS:
            n_splits = len(splits.load(kind))
            for fs in data.FEATURE_SETS:
                for s in range(n_splits):
                    path = UNC / "xgb" / method / kind / fs / f"split_{s:02d}.parquet"
                    if need(path, require_all) and need(path.with_suffix(".json"), require_all):
                        out[method, kind, fs, s] = pd.read_parquet(path)
    return out


def interval_stats(df) -> dict:
    out = {}
    for name, (lo, hi) in metrics.INTERVALS.items():
        iv = metrics.interval(df[f"q{lo:g}"], df[f"q{hi:g}"], df.y)
        out[f"coverage_{name}"], out[f"width_{name}"] = iv["coverage"], iv["width"]
    out["crps"] = float(df.crps.mean()) if "crps" in df else np.nan
    out["crps20"] = float(df.crps20.mean()) if "crps20" in df else np.nan
    out["rmse"] = metrics.point(df.y, df["mean"])["rmse"]
    return out


def per_split_table(dists, inputs):
    rows = []
    for (model, kind, fs, s), df in dists.items():
        r = df.row.to_numpy()
        groups = metrics.subgroup_masks(df.y, inputs.family[r], inputs.oxygen_missing[r])
        fold = splits.FAMILY_ORDER[s] if kind == "leave_family_out" else "all splits"
        for group, mask in groups.items():
            if mask.sum() >= 20:
                rows.append(
                    {
                        "model": model,
                        "kind": kind,
                        "features": fs,
                        "split": s,
                        "fold": fold,
                        "group": group,
                        "n": int(mask.sum()),
                        **interval_stats(df[mask]),
                    }
                )
    return pd.DataFrame(rows)


def coverage_curve(dists):
    rows = []
    pooled = {}
    for (model, kind, fs, _), df in dists.items():
        if kind != "leave_family_out":
            pooled.setdefault((model, kind, fs), []).append(df.pit.to_numpy())
    for (model, kind, fs), pits in pooled.items():
        pit = np.concatenate(pits)
        for level in CURVE_LEVELS:
            rows.append(
                {
                    "model": model,
                    "kind": kind,
                    "features": fs,
                    "nominal": level,
                    "empirical": float(np.mean(np.abs(pit - 0.5) <= level / 2 + 1e-12)),
                }
            )
    return pd.DataFrame(rows), pooled


def pit_hist(pooled):
    edges = np.linspace(0, 1, 21)
    rows = []
    for (model, kind, fs), pits in pooled.items():
        counts, _ = np.histogram(np.concatenate(pits), bins=edges)
        for lo, c in zip(edges[:-1], counts, strict=True):
            rows.append(
                {
                    "model": model,
                    "kind": kind,
                    "features": fs,
                    "bin_lo": lo,
                    "density": c / counts.sum() * 20,
                }
            )
    return pd.DataFrame(rows)


def p77_reliability(dists):
    edges = np.array([0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
    rows, brier = [], []
    pooled = {}
    for (model, kind, fs, _), df in dists.items():
        if kind != "leave_family_out":
            pooled.setdefault((model, kind, fs), []).append(df[["p_above_77K", "y"]])
    for (model, kind, fs), frames in pooled.items():
        d = pd.concat(frames)
        hit = (d.y > 77).to_numpy(float)
        p = d.p_above_77K.to_numpy(float)
        brier.append(
            {
                "model": model,
                "kind": kind,
                "features": fs,
                "brier": float(np.mean((p - hit) ** 2)),
                "base_rate_brier": float(np.mean((hit.mean() - hit) ** 2)),
            }
        )
        bins = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
        for b in np.unique(bins):
            m = bins == b
            rows.append(
                {
                    "model": model,
                    "kind": kind,
                    "features": fs,
                    "bin_lo": edges[b],
                    "bin_hi": edges[b + 1],
                    "n": int(m.sum()),
                    "predicted": float(p[m].mean()),
                    "observed": float(hit[m].mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(brier)


def lfo_table(dists, per_split, inputs):
    rows = []
    for (model, kind, fs, s), df in dists.items():
        if kind != "leave_family_out":
            continue
        family = splits.FAMILY_ORDER[s]
        width = (df["q0.975"] - df["q0.025"]).to_numpy()
        err = np.abs(df["mean"] - df.y).to_numpy()
        # Conformal intervals have one width for every row, so no rank correlation exists.
        # (float32 storage leaves rounding noise in those widths, hence the tolerance).
        constant = np.ptp(width) <= 1e-3 * max(float(np.mean(width)), 1e-9)
        rho = np.nan if constant else spearmanr(width, err).statistic
        held = interval_stats(df)
        ind = per_split[
            (per_split.model == model)
            & (per_split.kind == "grouped")
            & (per_split.features == fs)
            & (per_split.group == f"family: {family}")
        ]
        rec = {
            "model": model,
            "features": fs,
            "held_out_family": family,
            "n": len(df),
            "width_error_spearman": float(rho),
        }
        for k, v in held.items():
            rec[f"held_out_{k}"] = v
            rec[f"in_distribution_{k}"] = float(ind[k].mean()) if len(ind) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    inputs = benchmark.load_inputs()

    per_job = lc_jobs(inputs, args.require_all)
    curves, paired = lc_tables(per_job)
    LC.mkdir(parents=True, exist_ok=True)
    per_job.to_csv(LC / "per_job.csv", index=False, float_format="%.4f")
    curves.to_csv(LC / "curves.csv", index=False, float_format="%.4f")
    paired.to_csv(LC / "paired.csv", index=False, float_format="%.4f")
    (LC / "machines.json").write_text(json.dumps(lc_machines(), indent=1, sort_keys=True) + "\n")

    dists = load_distributions(args.require_all)
    per_split = per_split_table(dists, inputs)
    per_split.to_parquet(UNC / "per_split.parquet", compression="zstd", index=False)
    by = ["model", "kind", "features", "fold", "group"]
    cal = per_split.groupby(by, sort=False).mean(numeric_only=True).drop(columns="split")
    cal.reset_index().to_csv(UNC / "calibration.csv", index=False, float_format="%.4f")
    curve, pooled = coverage_curve(dists)
    curve.to_csv(UNC / "curve.csv", index=False, float_format="%.4f")
    pit_hist(pooled).to_csv(UNC / "pit_hist.csv", index=False, float_format="%.4f")
    rel, brier = p77_reliability(dists)
    rel.to_csv(UNC / "p77_reliability.csv", index=False, float_format="%.4f")
    brier.to_csv(UNC / "p77_brier.csv", index=False, float_format="%.5f")
    lfo_table(dists, per_split, inputs).to_csv(UNC / "lfo.csv", index=False, float_format="%.4f")
    models_seen = sorted({k[0] for k in dists})
    print(f"learning-curve jobs: {len(per_job)}; distribution sets: {len(dists)} ({models_seen})")


if __name__ == "__main__":
    main()
