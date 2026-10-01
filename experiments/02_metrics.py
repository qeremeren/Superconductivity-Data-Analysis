"""Phase 2 metrics from every model's committed predictions (no API calls).

Reads results/02_benchmark/predictions/<model>/<kind>/<features>.parquet and the
TabPFN per-split summaries in results/02_benchmark/tabpfn/<kind>/<features>/.
Writes to results/02_benchmark/:
  per_split_metrics.parquet  point metrics per model, split and row group
  metrics.csv                aggregated over splits (RMSE = sqrt(mean MSE), as in the paper);
                             leave-family-out folds are reported one by one
  paired.json                per-split comparisons of TabPFN against each other model
  replication.json           our random-split numbers next to Hamidieh (2018)
Run from the repo root: `uv run python -m experiments.02_metrics`.
"""

from __future__ import annotations

import json

import pandas as pd

from src import benchmark, metrics, splits

PAPER = {"xgb_hamidieh": {"rmse": 9.5, "r2": 0.92}, "linear": {"rmse": 17.6, "r2": 0.74}}


def load_predictions() -> pd.DataFrame:
    frames = []
    for path in sorted((benchmark.OUT / "predictions").glob("*/*/*.parquet")):
        model, kind, features = path.parts[-3], path.parts[-2], path.stem
        frames.append(pd.read_parquet(path).assign(model=model, kind=kind, features=features))
    for path in sorted((benchmark.OUT / "tabpfn").glob("*/*/split_*.parquet")):
        kind, features, split = path.parts[-3], path.parts[-2], int(path.stem.split("_")[1])
        df = pd.read_parquet(path, columns=["row", "mean"]).rename(columns={"mean": "pred"})
        frames.append(df.assign(split=split, model="tabpfn", kind=kind, features=features))
    return pd.concat(frames, ignore_index=True)


def per_split_metrics(preds: pd.DataFrame, inputs) -> pd.DataFrame:
    records = []
    for (model, kind, features, split), df in preds.groupby(
        ["model", "kind", "features", "split"], sort=False
    ):
        rows = df["row"].to_numpy()
        y = inputs.y[rows]
        groups = metrics.subgroup_masks(y, inputs.family[rows], inputs.oxygen_missing[rows])
        fold = splits.FAMILY_ORDER[split] if kind == "leave_family_out" else "all splits"
        for group, mask in groups.items():
            if mask.sum() < 2:
                continue
            m = metrics.point(y[mask], df["pred"].to_numpy()[mask])
            records.append(
                {
                    "model": model,
                    "kind": kind,
                    "features": features,
                    "split": split,
                    "fold": fold,
                    "group": group,
                    **m,
                }
            )
    return pd.DataFrame(records)


def paired_vs_tabpfn(per_split: pd.DataFrame) -> list[dict]:
    rmse = per_split[per_split.group == "all"].set_index(["model", "kind", "features", "split"])
    rmse = rmse["rmse"]
    out = []
    if "tabpfn" not in rmse.index.get_level_values("model"):
        return out
    for (kind, features), _ in rmse.groupby(level=["kind", "features"]):
        if kind == "leave_family_out":
            continue
        tab = rmse.xs(("tabpfn", kind, features), level=["model", "kind", "features"])
        for other in sorted(set(rmse.index.get_level_values("model")) - {"tabpfn"}):
            try:
                b = rmse.xs((other, kind, features), level=["model", "kind", "features"])
            except KeyError:
                continue
            out.append(
                {
                    "kind": kind,
                    "features": features,
                    "a": "tabpfn",
                    "b": other,
                    "metric": "rmse",
                    **metrics.paired(tab, b),
                }
            )
    return out


def main():
    inputs = benchmark.load_inputs()
    preds = load_predictions()
    per_split = per_split_metrics(preds, inputs)
    per_split.to_parquet(benchmark.OUT / "per_split_metrics.parquet", compression="zstd")

    table = metrics.aggregate(per_split, ["model", "kind", "features", "fold", "group"])
    table.to_csv(benchmark.OUT / "metrics.csv", index=False, float_format="%.4f")
    (benchmark.OUT / "paired.json").write_text(json.dumps(paired_vs_tabpfn(per_split), indent=1))

    rep = table[(table.kind == "random") & (table.group == "all")]
    replication = {
        model: {
            "paper": PAPER.get(model),
            **{
                fs: {"rmse": float(r.rmse), "r2": float(r.r2), "n_splits": int(r.n_splits)}
                for fs, r in rep[rep.model == model].set_index("features").iterrows()
            },
        }
        for model in sorted(rep.model.unique())
    }
    (benchmark.OUT / "replication.json").write_text(json.dumps(replication, indent=1) + "\n")

    show = table[table.group == "all"][
        ["model", "kind", "features", "fold", "n_splits", "rmse", "rmse_split_std", "mae", "r2"]
    ]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(replication, indent=1))


if __name__ == "__main__":
    main()
