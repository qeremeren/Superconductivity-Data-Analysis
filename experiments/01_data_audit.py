"""Phase 1 data audit. No API calls.

Writes results/01_audit/:
  summary.json    every number reported from this phase
  duplicates.csv  one row per scaled composition that occurs more than once
  pool.csv        the deduplicated discovery pool (median Tc per composition)
  nn_split0.csv   per test row nearest-train-composition data for split 0 of
                  each split kind (feeds the figures)

Run from the repo root: `uv run python -m experiments.01_data_audit`.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src import config, data, splits

OUT_DIR = config.RESULTS_DIR / "01_audit"
T77 = 77.0
TOP_FRACTION = 0.01
NN_THRESHOLDS = (0.01, 0.02, 0.05, 0.1)
NN_CHUNK = 1000
FAMILIES = (data.CUPRATE, data.IRON_BASED, data.OTHER)
SIZE_BINS = [0, 1, 2, 5, 10, 50, np.inf]
SIZE_LABELS = ["1", "2", "3-5", "6-10", "11-50", ">50"]


def rmse(err) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def quantiles(x) -> dict:
    q = np.quantile(np.asarray(x, dtype=float), [0, 0.25, 0.5, 0.75, 1])
    return dict(zip(["min", "q25", "median", "q75", "max"], map(float, q), strict=True))


def alignment(train, um) -> dict:
    same_tc = train[data.TARGET].to_numpy() == um[data.TARGET].to_numpy()
    n_nonzero = (um[list(data.ELEMENTS)] > 0).sum(axis=1).to_numpy()
    same_n = train["number_of_elements"].to_numpy() == n_nonzero
    return {
        "rows_train_csv": len(train),
        "rows_unique_m_csv": len(um),
        "same_row_count": len(train) == len(um),
        "critical_temp_identical_rows": int(same_tc.sum()),
        "critical_temp_identical_all_rows": bool(same_tc.all()),
        "number_of_elements_matches_rows": int(same_n.sum()),
    }


def overview(um, key) -> dict:
    tc = um[data.TARGET]
    raw_vectors = pd.Series([r.tobytes() for r in um[list(data.ELEMENTS)].round(6).to_numpy()])
    spellings = um.groupby(key)[data.FORMULA].nunique()
    return {
        "rows": len(um),
        "tc_K": quantiles(tc),
        "rows_tc_below_1K": int((tc < 1).sum()),
        "unique_formula_strings": int(um[data.FORMULA].nunique()),
        "unique_raw_composition_vectors": int(raw_vectors.nunique()),
        "unique_scaled_compositions": int(key.nunique()),
        "scaled_compositions_with_several_spellings": int((spellings > 1).sum()),
    }


def by_family(um, key, fam) -> dict:
    tc = um[data.TARGET]
    out = {}
    for name in FAMILIES:
        m = fam == name
        out[name] = {
            "rows": int(m.sum()),
            "materials": int(key[m].nunique()),
            "tc_K": quantiles(tc[m]),
            **{f"rows_above_{t}K": int((tc[m] > t).sum()) for t in (77, 100, 120)},
        }
    return out


def duplicates(um, key, fam) -> tuple[dict, pd.DataFrame]:
    tc = um[data.TARGET]
    g = pd.DataFrame({"key": key, "tc": tc, "family": fam, "formula": um[data.FORMULA]})
    grp = g.groupby("key")
    table = pd.DataFrame(
        {
            "family": grp["family"].first(),
            "n_rows": grp.size(),
            "n_formula_strings": grp["formula"].nunique(),
            "formulas": grp["formula"].agg(lambda s: "; ".join(pd.unique(s)[:3])),
            "tc_min": grp["tc"].min(),
            "tc_median": grp["tc"].median(),
            "tc_max": grp["tc"].max(),
            "tc_std": grp["tc"].std(),
        }
    )
    dup = table[table.n_rows > 1].sort_values("n_rows", ascending=False)

    in_dup = g["key"].isin(dup.index)
    resid = g.loc[in_dup, "tc"] - grp["tc"].transform("mean")[in_dup]
    pooled_sd = math.sqrt(float((resid**2).sum()) / (in_dup.sum() - len(dup)))

    # Memorization: predict each duplicated row by the mean Tc of the other rows
    # with the same composition (leave-one-out within the group).
    n = grp["tc"].transform("size")[in_dup]
    total = grp["tc"].transform("sum")[in_dup]
    loo_pred = (total - g.loc[in_dup, "tc"]) / (n - 1)
    loo_err = loo_pred - g.loc[in_dup, "tc"]

    sizes = table.n_rows
    bins = pd.cut(sizes, SIZE_BINS, labels=SIZE_LABELS).value_counts(sort=False)
    largest = dup.iloc[0]
    summary = {
        "materials": len(table),
        "materials_with_duplicates": len(dup),
        "rows_in_duplicated_materials": int(in_dup.sum()),
        "materials_by_group_size": {str(k): int(v) for k, v in bins.items()},
        "largest_group": {"formulas": largest.formulas, "n_rows": int(largest.n_rows)},
        "within_material_tc_sd_pooled_K": pooled_sd,
        "within_material_tc_sd_median_K": float(dup.tc_std.median()),
        "within_material_tc_range_max_K": float((dup.tc_max - dup.tc_min).max()),
        "loo_duplicate_mean_rmse_K": rmse(loo_err),
        "loo_duplicate_mean_mae_K": float(np.mean(np.abs(loo_err))),
        # Every model here is a function of composition (see features), so none can
        # beat predicting each material's mean Tc: this RMSE over all rows is a floor.
        "rmse_floor_any_composition_model_K": math.sqrt(float((resid**2).sum()) / len(g)),
    }
    return summary, dup


def nonoxygen_key(um) -> pd.Series:
    """Composition with oxygen removed and the rest rescaled: two materials with the
    same key differ only in oxygen content (e.g. YBa2Cu3O6.9 vs YBa2Cu3O7)."""
    return data.composition_key(um[list(data.ELEMENTS)].assign(O=0.0))


def split_leakage(fractions, key, nonox, tc, fam, masks, kind) -> tuple[dict, pd.DataFrame]:
    """Nearest train composition (L1 on element fractions) for every test row."""
    key, nonox, fam = key.to_numpy(), nonox.to_numpy(), fam.to_numpy()
    tc = np.asarray(tc, dtype=float)
    first_row = pd.Series(np.arange(len(key))).groupby(key).first()  # key -> a row index
    per_split, rows0 = [], None
    for s, is_test in enumerate(masks):
        train_keys = pd.Series(tc[~is_test]).groupby(key[~is_test]).median()
        train_vecs = fractions[first_row.loc[train_keys.index].to_numpy()]
        test_keys, test_inverse = np.unique(key[is_test], return_inverse=True)
        test_vecs = fractions[first_row.loc[test_keys].to_numpy()]

        nn_dist = np.empty(len(test_keys))
        nn_idx = np.empty(len(test_keys), dtype=int)
        for start in range(0, len(test_keys), NN_CHUNK):
            d = cdist(test_vecs[start : start + NN_CHUNK], train_vecs, metric="cityblock")
            nn_idx[start : start + NN_CHUNK] = d.argmin(axis=1)
            nn_dist[start : start + NN_CHUNK] = d.min(axis=1)

        dist = nn_dist[test_inverse]
        nn_tc = train_keys.to_numpy()[nn_idx][test_inverse]
        exact_in_train = np.isin(key[is_test], train_keys.index)
        train_full = set(key[~is_test])
        train_nonox = pd.Series(key[~is_test]).groupby(nonox[~is_test]).agg(set)
        oxy_variant = np.array(
            [
                nk in train_nonox.index and bool(train_nonox[nk] - {k})
                for k, nk in zip(key[is_test], nonox[is_test], strict=True)
            ]
        )
        assert kind != "grouped" or not (set(key[is_test]) & train_full)

        rec = {
            "exact_duplicate_in_train": float(exact_in_train.mean()),
            "oxygen_only_variant_in_train": float(oxy_variant.mean()),
            "nn_distance_median": float(np.median(dist)),
            **{f"nn_distance_below_{t}": float((dist < t).mean()) for t in NN_THRESHOLDS},
            "one_nn_mse": float(np.mean((nn_tc - tc[is_test]) ** 2)),
        }
        test_fam = fam[is_test]
        for name in FAMILIES:
            m = test_fam == name
            rec[f"{name}:nn_distance_below_0.02"] = float((dist[m] < 0.02).mean())
            rec[f"{name}:oxygen_only_variant_in_train"] = float(oxy_variant[m].mean())
        per_split.append(rec)
        if s == 0:
            rows0 = pd.DataFrame(
                {
                    "row": np.flatnonzero(is_test),
                    "family": test_fam,
                    "tc": tc[is_test],
                    "nn_distance": dist,
                    "nn_tc": nn_tc,
                    "exact_duplicate_in_train": exact_in_train,
                    "oxygen_only_variant_in_train": oxy_variant,
                }
            )

    df = pd.DataFrame(per_split)
    summary = {c: float(df[c].mean()) for c in df.columns if c != "one_nn_mse"}
    summary["one_nn_rmse_K"] = float(np.sqrt(df["one_nn_mse"].mean()))  # paper-style RMSE
    summary["n_splits"] = len(df)
    return summary, rows0.assign(split_kind=kind)


def target_stats(tc, threshold) -> dict:
    k = int((tc >= threshold).sum())
    n = len(tc)
    return {
        "threshold_K": float(threshold),
        "hits": k,
        "base_rate": k / n,
        "random_expected_tries_to_first_hit": (n + 1) / (k + 1),
    }


def suspect_reasons(um, fam) -> pd.Series:
    """Rule-based flags for entries to review before the discovery pools use them.
    Nothing is removed here; the pool reports its targets with and without them."""
    tc = um[data.TARGET]
    no_cu_oxide = (fam == data.OTHER) & (um["O"] > 0) & ((um["Ba"] > 0) | (um["Sr"] > 0))
    no_cu_oxide &= um["Cu"] == 0
    rules = {
        "cuprate-like oxide without Cu, Tc > 40 K": no_cu_oxide & (tc > 40),
        "non-cuprate above 77 K": (fam != data.CUPRATE) & (tc > T77),
    }
    return pd.Series(
        ["; ".join(r for r, m in rules.items() if m[i]) for i in um.index], index=um.index
    )


def discovery_pool(um, key, fam) -> tuple[dict, pd.DataFrame]:
    tc = um[data.TARGET]
    reasons = suspect_reasons(um, fam)
    g = pd.DataFrame(
        {"key": key, "tc": tc, "family": fam, "formula": um[data.FORMULA], "why": reasons}
    )
    grp = g.groupby("key")
    pool = pd.DataFrame(
        {
            "family": grp["family"].first(),
            "tc_median": grp["tc"].median(),
            "n_rows": grp.size(),
            "example_formula": grp["formula"].first(),
            "suspect": grp["why"].agg(lambda s: "; ".join(sorted({r for r in s if r}))),
        }
    )
    non_cuprate = pool[pool.family != data.CUPRATE]
    out = {
        "suspect_materials": pool.loc[
            pool.suspect != "", ["example_formula", "family", "tc_median", "suspect"]
        ]
        .sort_values("tc_median", ascending=False)
        .to_dict(orient="records")
    }
    pools = (
        ("all", pool),
        ("non_cuprate", non_cuprate),
        ("non_cuprate_excluding_suspects", non_cuprate[non_cuprate.suspect == ""]),
    )
    for name, sub in pools:
        tc = sub.tc_median.to_numpy()
        k = math.ceil(TOP_FRACTION * len(tc))
        top_threshold = np.sort(tc)[::-1][k - 1]
        above77 = tc > T77
        top = sub[sub.tc_median >= top_threshold]
        out[name] = {
            "materials": len(sub),
            "above_77K": {
                "hits": int(above77.sum()),
                "base_rate": float(above77.mean()),
                "random_expected_tries_to_first_hit": (len(tc) + 1) / (above77.sum() + 1),
            },
            "top_1pct": {
                **target_stats(tc, top_threshold),
                "hits_by_family": top.family.value_counts().to_dict(),
                "hit_tc_range_K": [float(top.tc_median.min()), float(top.tc_median.max())],
            },
        }
        pool[f"top1_{name}"] = pool.index.isin(top.index)
    return out, pool


def modal_share(x) -> dict:
    rounded = np.round(np.asarray(x, dtype=float), 3)
    values, counts = np.unique(rounded, return_counts=True)
    return {"value": float(values[counts.argmax()]), "share": float(counts.max() / len(rounded))}


def feature_checks(train, key, fam) -> dict:
    feats = train.drop(columns=[data.TARGET])
    grp = feats.groupby(key.to_numpy())
    spread = grp.max() - grp.min()
    inconsistent = (spread > 1e-6).any(axis=1)
    constant = [c for c in feats.columns if feats[c].nunique() == 1]
    dup_cols = feats.T.duplicated()
    rtc = train["range_ThermalConductivity"]
    return {
        "materials_with_differing_features": int(inconsistent.sum()),
        "max_feature_spread_within_material": float(spread.to_numpy().max()),
        "constant_features": constant,
        "duplicated_feature_columns": list(feats.columns[dup_cols.to_numpy()]),
        "range_ThermalConductivity_by_family": {
            name: quantiles(rtc[fam == name]) for name in FAMILIES
        },
        "range_ThermalConductivity_mode_by_family": {
            name: modal_share(rtc[fam == name]) for name in FAMILIES
        },
    }


def oxygen_stoichiometry(um, key, fam) -> dict:
    cup = um[fam == data.CUPRATE]
    integer = np.isclose(cup["O"], np.round(cup["O"]), atol=1e-9)
    mats = pd.Series(integer, index=cup.index).groupby(key[fam == data.CUPRATE]).first()
    # "O" followed by neither an amount nor a lowercase letter (Os): the source gave
    # no oxygen content, and unique_m.csv encodes it as O = 1.
    bare_o = um[data.FORMULA].str.contains(r"O(?![a-z\d.])", regex=True)
    tc = um[data.TARGET]
    return {
        "rows_oxygen_amount_missing": int(bare_o.sum()),
        "rows_oxygen_amount_missing_encoded_as": sorted(map(float, um.loc[bare_o, "O"].unique())),
        "cuprate_rows_oxygen_amount_missing": int((bare_o & (fam == data.CUPRATE)).sum()),
        "cuprate_median_tc_oxygen_missing_K": float(tc[bare_o & (fam == data.CUPRATE)].median()),
        "cuprate_median_tc_oxygen_given_K": float(tc[~bare_o & (fam == data.CUPRATE)].median()),
        "cuprate_rows_integer_O": int(integer.sum()),
        "cuprate_rows": len(cup),
        "cuprate_rows_integer_O_share": float(integer.mean()),
        "cuprate_materials_integer_O_share": float(mats.mean()),
    }


def l1_example() -> dict:
    """Distance scale for interpreting NN thresholds: YBa2Cu3O7 vs YBa2Cu3O6.9."""
    a = {"Y": 1, "Ba": 2, "Cu": 3, "O": 7}
    b = {**a, "O": 6.9}
    va, vb = (np.array([d.get(e, 0) for e in data.ELEMENTS], float) for d in (a, b))
    return {"YBa2Cu3O7_vs_YBa2Cu3O6.9": float(np.abs(va / va.sum() - vb / vb.sum()).sum())}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, um = data.load_train(), data.load_unique_m()
    key, fam = data.composition_key(um), data.family(um)
    fractions = data.composition_fractions(um).to_numpy()
    nonox = nonoxygen_key(um)

    summary = {"alignment": alignment(train, um), "overview": overview(um, key)}
    summary["families"] = by_family(um, key, fam)
    summary["duplicates"], dup = duplicates(um, key, fam)
    summary["leakage"] = {"l1_distance_example": l1_example()}
    nn_rows = []
    for kind in ("random", "grouped"):
        stats, rows = split_leakage(
            fractions, key, nonox, um[data.TARGET], fam, splits.load(kind), kind
        )
        summary["leakage"][kind] = stats
        nn_rows.append(rows)
    summary["discovery_pool"], pool = discovery_pool(um, key, fam)
    summary["features"] = feature_checks(train, key, fam)
    summary["oxygen"] = oxygen_stoichiometry(um, key, fam)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    dup.to_csv(OUT_DIR / "duplicates.csv", index_label="composition_key", float_format="%.6g")
    pool.to_csv(OUT_DIR / "pool.csv", index_label="composition_key", float_format="%.6g")
    pd.concat(nn_rows).to_csv(OUT_DIR / "nn_split0.csv", index=False, float_format="%.6g")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
