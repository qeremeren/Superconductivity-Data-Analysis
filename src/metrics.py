"""Point and distributional metrics, and how they are aggregated over splits.

Aggregation follows Hamidieh (2018): RMSE over repeated splits is
sqrt(mean of per-split MSEs), not the mean of per-split RMSEs.

Distributional metrics work from a quantile grid. QUANTILE_LEVELS is the 100-level
midpoint grid tau_i = (i - 0.5)/100, on which expectations over the predictive
distribution are a midpoint rule, plus the levels needed for 50/80/90/95%
intervals. CRPS uses the midpoint levels only; PIT and P(Tc > t) interpolate the
quantile function over all levels and are therefore clamped to [0.005, 0.995].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIDPOINT_LEVELS = np.round((np.arange(100) + 0.5) / 100, 3)
INTERVAL_LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
QUANTILE_LEVELS = np.unique(np.concatenate([MIDPOINT_LEVELS, INTERVAL_LEVELS]))
# A coarser grid for models that fit one output per quantile (XGBoost quantile regression).
# Its 20 midpoints (0.025, 0.075, ..., 0.975) are all in MIDPOINT_LEVELS, so TabPFN's CRPS can
# be computed on exactly the same levels for a like-for-like comparison.
MIDPOINT_20 = np.round((np.arange(20) + 0.5) / 20, 3)
LEVELS_27 = np.unique(np.concatenate([MIDPOINT_20, INTERVAL_LEVELS]))
# Quantiles kept in the committed per-row summaries.
SUMMARY_LEVELS = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)
INTERVALS = {"50%": (0.25, 0.75), "80%": (0.1, 0.9), "90%": (0.05, 0.95), "95%": (0.025, 0.975)}

TC_BANDS = {"Tc < 10 K": (-np.inf, 10.0), "10-77 K": (10.0, 77.0), "Tc > 77 K": (77.0, np.inf)}


def point(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    mse = float(np.mean(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(y),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err**2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def band_masks(y) -> dict[str, np.ndarray]:
    """Bands by true Tc: below 10 K, 10 to 77 K inclusive, above 77 K."""
    y = np.asarray(y, float)
    return {
        "Tc < 10 K": y < 10,
        "10-77 K": (y >= 10) & (y <= 77),
        "Tc > 77 K": y > 77,
    }


def subgroup_masks(y, family, oxygen_missing) -> dict[str, np.ndarray]:
    """Every row group a metric is reported on: all rows, Tc bands, families, and rows
    whose formula gives no oxygen amount."""
    family = np.asarray(family)
    groups = {"all": np.ones(len(family), dtype=bool), **band_masks(y)}
    for name in ("cuprate", "iron-based", "other"):
        groups[f"family: {name}"] = family == name
    groups["oxygen amount missing"] = np.asarray(oxygen_missing, dtype=bool)
    return groups


def aggregate(per_split: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Combine per-split point metrics (columns n, mse, mae, r2) over splits.

    rmse = sqrt(mean mse) as in the paper; mae and r2 are mean and std over splits;
    rmse_split_std is the spread of the per-split RMSEs.
    """
    g = per_split.groupby(by, sort=False)
    out = pd.DataFrame(
        {
            "n_splits": g.size(),
            "rows_per_split": g["n"].mean(),
            "rmse": np.sqrt(g["mse"].mean()),
            "rmse_split_std": g["mse"].apply(
                lambda m: float(np.std(np.sqrt(m), ddof=1)) if len(m) > 1 else np.nan
            ),
            "mae": g["mae"].mean(),
            "mae_std": g["mae"].std(),
            "r2": g["r2"].mean(),
            "r2_std": g["r2"].std(),
        }
    )
    return out.reset_index()


def paired(a: pd.Series, b: pd.Series) -> dict:
    """Per-split comparison of a metric where lower is better (e.g. RMSE): a vs b,
    aligned on the split index."""
    a, b = a.align(b, join="inner")
    diff = a - b
    return {
        "n_splits": len(diff),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
        "a_better_splits": int((diff < 0).sum()),
        "b_better_splits": int((diff > 0).sum()),
    }


def crps(quantiles: np.ndarray, levels: np.ndarray, y, midpoints=MIDPOINT_LEVELS) -> np.ndarray:
    """CRPS per row from quantiles on a midpoint grid: 2 * mean of pinball losses.

    quantiles: (n, len(levels)); only the columns at `midpoints` are used.
    """
    cols = np.isin(np.round(levels, 3), midpoints)
    if cols.sum() != len(midpoints):
        raise ValueError("quantiles must include every midpoint level")
    q, tau = quantiles[:, cols], np.asarray(levels)[cols]
    y = np.asarray(y, float)[:, None]
    pinball = ((y < q).astype(float) - tau) * (q - y)
    return 2 * pinball.mean(axis=1)


def cdf_at(quantiles: np.ndarray, levels: np.ndarray, x) -> np.ndarray:
    """Predictive CDF at x per row, by linear interpolation of the quantile function.
    Values beyond the outermost quantiles are clamped to the outermost levels."""
    levels = np.asarray(levels, float)
    x = np.broadcast_to(np.asarray(x, float), (len(quantiles),))
    out = np.empty(len(quantiles))
    for i, (q, xi) in enumerate(zip(quantiles, x, strict=True)):
        q_sorted = np.maximum.accumulate(q)  # guard against tiny non-monotonicity
        out[i] = np.interp(xi, q_sorted, levels)
    return out


def interval(lo, hi, y) -> dict:
    y = np.asarray(y, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "width": float(np.mean(hi - lo))}


def summarize_distribution(
    mean, quantiles: np.ndarray, levels: np.ndarray, y, midpoints=MIDPOINT_LEVELS
) -> pd.DataFrame:
    """The per-row summary committed for distributional models: mean, quantiles at
    SUMMARY_LEVELS, CRPS (on `midpoints`), PIT and P(Tc > 77 K)."""
    levels = np.round(np.asarray(levels, float), 3)
    out = {"mean": np.asarray(mean, float)}
    for lvl in SUMMARY_LEVELS:
        (col,) = np.flatnonzero(levels == lvl)
        out[f"q{lvl:g}"] = quantiles[:, col]
    out["crps"] = crps(quantiles, levels, y, midpoints)
    out["pit"] = cdf_at(quantiles, levels, y)
    out["p_above_77K"] = 1 - cdf_at(quantiles, levels, 77.0)
    return pd.DataFrame(out).astype("float32")


# --- exact metrics for TabPFN's full output (a piecewise-uniform "bar" distribution) ----


def bar_cdf_at_borders(logits: np.ndarray, borders: np.ndarray) -> np.ndarray:
    """CDF at every bucket border, shape (n, n_buckets + 1), from per-bucket logits."""
    z = np.asarray(logits, float)
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return np.concatenate([np.zeros((len(p), 1)), np.cumsum(p, axis=1)], axis=1)


def bar_cdf(logits, borders, x) -> np.ndarray:
    """Exact CDF at x per row (linear within each bucket)."""
    cdf = bar_cdf_at_borders(logits, borders)
    x = np.broadcast_to(np.asarray(x, float), (len(cdf),))
    return np.array([np.interp(xi, borders, c) for xi, c in zip(x, cdf, strict=True)])


def bar_quantiles(logits, borders, levels) -> np.ndarray:
    """Exact quantiles per row, shape (n, len(levels)), by inverting the CDF."""
    cdf = bar_cdf_at_borders(logits, borders)
    return np.stack([np.interp(levels, c, borders) for c in cdf])


def bar_crps(logits, borders, y) -> np.ndarray:
    """Exact CRPS per row: integral of (F(x) - 1[x >= y])^2. F is linear within each
    bucket, so each piece integrates to L/3 (u^2 + uv + v^2) for end values u, v."""
    borders = np.asarray(borders, float)
    cdf = bar_cdf_at_borders(logits, borders)
    widths = np.diff(borders)
    out = np.empty(len(cdf))
    for i, (c, yi) in enumerate(zip(cdf, np.asarray(y, float), strict=True)):
        j = int(np.clip(np.searchsorted(borders, yi, side="right") - 1, 0, len(widths) - 1))
        lo, hi = c[:-1], c[1:]
        below = widths[:j] / 3 * (lo[:j] ** 2 + lo[:j] * hi[:j] + hi[:j] ** 2)
        a, b = 1 - lo[j + 1 :], 1 - hi[j + 1 :]
        above = widths[j + 1 :] / 3 * (a**2 + a * b + b**2)
        fy = np.interp(yi, borders[j : j + 2], c[j : j + 2])
        left = (yi - borders[j]) / 3 * (c[j] ** 2 + c[j] * fy + fy**2)
        u, v = 1 - fy, 1 - c[j + 1]
        right = (borders[j + 1] - yi) / 3 * (u**2 + u * v + v**2)
        out[i] = below.sum() + above.sum() + left + right
    return out
