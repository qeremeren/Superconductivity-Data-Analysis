"""Phase 5: is Hamidieh's top feature, range_ThermalConductivity, a cuprate detector?

Hamidieh (2018) found range_ThermalConductivity carries ~30% of the published XGBoost's gain.
Phase 1 found that 98.5% of cuprate rows share one value of it (Cu minus O). Two tests:
  1. Without fitting: how well the feature alone separates cuprates from the rest.
  2. Refit the published XGBoost (engineered features) on grouped splits 0-4 as in Phase 2,
     with and without an explicit is_cuprate column; compare gain shares and test RMSE. The
     refit without the column must reproduce the committed Phase 2 predictions.
Writes results/05_why/cuprate_detector.json and results/05_why/feature_gain.csv. Fits XGBoost
locally (a few minutes, no API key); `make why-xgb`.

Run from the repo root: `uv run python -m experiments.05_cuprate_detector`.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import benchmark, config, data, models

OUT = config.RESULTS_DIR / "05_why"
FEATURE = "range_ThermalConductivity"
SPLITS = (0, 1, 2, 3, 4)


def gain_shares(model, names) -> pd.Series:
    score = model.get_booster().get_score(importance_type="total_gain")
    gains = pd.Series({names[int(k[1:])]: v for k, v in score.items()}).reindex(names).fillna(0)
    return gains / gains.sum()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = benchmark.load_inputs()
    X = inputs.features["engineered"]
    cuprate = inputs.family == data.CUPRATE
    value = X[FEATURE].to_numpy()
    mode = pd.Series(np.round(value[cuprate], 3)).mode().iloc[0]
    at_mode = np.isclose(value, mode, atol=1e-3)
    report = {
        "feature": FEATURE,
        "alone": {
            "roc_auc_for_cuprate": float(roc_auc_score(cuprate, value)),
            "most_common_value_among_cuprates": float(mode),
            "share_cuprate_rows_at_that_value": float(at_mode[cuprate].mean()),
            "share_non_cuprate_rows_at_that_value": float(at_mode[~cuprate].mean()),
            "share_rows_at_that_value_that_are_cuprates": float(cuprate[at_mode].mean()),
        },
    }
    committed = pd.read_parquet(benchmark.predictions_path("xgb_hamidieh", "grouped", "engineered"))
    rows, fits = [], []
    t0 = time.perf_counter()
    for _kind, i, seed, is_test in benchmark.iter_splits(("grouped",)):
        if i not in SPLITS:
            continue
        y_train, y_test = inputs.y[~is_test], inputs.y[is_test]
        variants = {
            "published": X,
            "published + is_cuprate": X.assign(is_cuprate=cuprate.astype(float)),
        }
        for name, Xv in variants.items():
            A = Xv.to_numpy(np.float32)
            m = models.hamidieh_xgb(seed).fit(A[~is_test], y_train)
            pred = m.predict(A[is_test])
            share = gain_shares(m, list(Xv.columns))
            fit = {
                "split": i,
                "variant": name,
                "rmse": float(np.sqrt(np.mean((pred - y_test) ** 2))),
                "gain_share_range_ThermalConductivity": float(share[FEATURE]),
                "gain_share_is_cuprate": float(share.get("is_cuprate", np.nan)),
                "top5": [[k, round(float(v), 4)] for k, v in share.nlargest(5).items()],
            }
            if name == "published":
                ref = committed[committed.split == i].sort_values("row").pred.to_numpy()
                fit["max_abs_diff_vs_committed_phase2_predictions"] = float(
                    np.abs(ref - pred).max()
                )
            fits.append(fit)
            rows += [
                {"split": i, "variant": name, "feature": k, "gain_share": float(v)}
                for k, v in share.items()
            ]
            print(
                f"split {i} {name}: rmse {fit['rmse']:.3f}, "
                f"{FEATURE} {fit['gain_share_range_ThermalConductivity']:.3f}, "
                f"is_cuprate {fit['gain_share_is_cuprate']:.3f}",
                flush=True,
            )
    f = pd.DataFrame(fits)
    report["refits"] = fits
    report["summary"] = {
        v: {
            "splits": int((f.variant == v).sum()),
            "rmse_mean": float(f[f.variant == v].rmse.mean()),
            "gain_share_range_ThermalConductivity_mean": float(
                f[f.variant == v].gain_share_range_ThermalConductivity.mean()
            ),
            "gain_share_is_cuprate_mean": float(f[f.variant == v].gain_share_is_cuprate.mean()),
        }
        for v in ("published", "published + is_cuprate")
    }
    report["machine"] = benchmark.machine_info()
    report["seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / "cuprate_detector.json").write_text(json.dumps(report, indent=1) + "\n")
    g = pd.DataFrame(rows).groupby(["variant", "feature"]).gain_share.mean().reset_index()
    g.sort_values(["variant", "gain_share"], ascending=[True, False]).to_csv(
        OUT / "feature_gain.csv", index=False, float_format="%.5f"
    )
    print(json.dumps({k: report[k] for k in ("alone", "summary")}, indent=1))


if __name__ == "__main__":
    main()
