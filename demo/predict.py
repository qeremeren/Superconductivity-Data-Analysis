"""Formula -> Tc with TabPFN-3.5: predictive mean, 80% and 95% intervals, P(Tc > 77 K),
and the nearest material in the training data.

    uv run python -m demo.predict HgBa2Ca2Cu3O8 MgB2          # from the committed cache
    uv run python -m demo.predict "La3Ni2O7" --live           # new formula: one API request

TabPFN is fit on all 21,263 rows of the data with composition features (86 element
fractions; the "TabPFN, composition" model of the benchmarks), seed 0. If the formula's
material (same scaled composition, any spelling) is in the data, its rows are held out of
training, so the prediction is what the model would say had it never seen that material.
Oxygen variants and other close relatives stay in, as in the grouped benchmark split; the
nearest training material and its distance are printed so you can judge how far the
prediction reaches.

Every prediction is cached under results/06_demo/predictions/ with a fingerprint of its
inputs. Without --live (or TABPFN_LIVE=1) the demo only reads the cache and never calls the
API. Phase 5 found that the interval width tracks how much Tc varies among similar training
materials, not how novel a formula is, so read the distance as a separate warning.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version

import numpy as np
import pandas as pd

from src import budget, config, data, metrics, models

OUT = config.RESULTS_DIR / "06_demo"
CACHE = OUT / "predictions"
EXPERIMENT = "p6_demo"
SEED = 0
LEVELS = metrics.QUANTILE_LEVELS
P_FLOOR = float(LEVELS[0])  # P(Tc > t) from the grid is clamped to [0.005, 0.995]

_TOKEN = re.compile(r"([A-Z][a-z]?|\(|\))(\d*\.?\d*)")


def parse_formula(formula: str) -> dict[str, float]:
    """Element amounts from a formula such as "HgBa2Ca2Cu3O8" or "(Ba0.6K0.4)Fe2As2"."""
    text = formula.replace(" ", "")
    stack: list[dict[str, float]] = [{}]
    pos = 0
    for m in _TOKEN.finditer(text):
        if m.start() != pos:
            raise ValueError(f"cannot parse {formula!r} at {text[pos:]!r}")
        pos = m.end()
        sym, num = m.groups()
        n = float(num) if num else 1.0
        if sym == "(":
            if num:
                raise ValueError(f"cannot parse {formula!r}: number after '('")
            stack.append({})
        elif sym == ")":
            if len(stack) == 1:
                raise ValueError(f"unbalanced ')' in {formula!r}")
            group = stack.pop()
            for el, v in group.items():
                stack[-1][el] = stack[-1].get(el, 0.0) + v * n
        else:
            if sym not in data.ELEMENTS:
                raise ValueError(
                    f"{sym!r} in {formula!r} is not one of the data's 86 elements (H to Rn)"
                )
            stack[-1][sym] = stack[-1].get(sym, 0.0) + n
    if pos != len(text) or len(stack) != 1 or not stack[0]:
        raise ValueError(f"cannot parse {formula!r}")
    if any(v <= 0 for v in stack[0].values()):
        raise ValueError(f"amounts in {formula!r} must be positive")
    return stack[0]


def composition_row(formula: str) -> pd.DataFrame:
    """One row in unique_m.csv's layout (element counts, critical_temp, material)."""
    counts = dict.fromkeys(data.ELEMENTS, 0.0) | parse_formula(formula)
    row = pd.DataFrame([counts], columns=list(data.ELEMENTS))
    return row.assign(**{data.TARGET: np.nan, data.FORMULA: formula})


@dataclass(frozen=True)
class Query:
    formula: str
    key: str
    family: str
    x: pd.DataFrame  # 1 x 86 element fractions
    X_train: pd.DataFrame
    y_train: np.ndarray
    held_out: pd.DataFrame  # the material's own rows in the data (empty if new)
    nearest: dict


def build_query(formula: str, unique_m: pd.DataFrame, keys: pd.Series) -> Query:
    row = composition_row(formula)
    key = data.composition_key(row).iloc[0]
    fractions = data.composition_fractions(unique_m)
    own = (keys == key).to_numpy()
    X_train = fractions[~own].reset_index(drop=True)
    y_train = unique_m.loc[~own, data.TARGET].to_numpy(np.float64)

    # Nearest training material (L1 over element fractions, one entry per material).
    train = pd.DataFrame(
        {"key": keys[~own].to_numpy(), "tc": y_train, "formula": unique_m.loc[~own, data.FORMULA]}
    )
    per_material = train.groupby("key").agg(tc=("tc", "median"), formula=("formula", "first"))
    material_x = (fractions[~own].assign(key=keys[~own].to_numpy()).groupby("key").first()).loc[
        per_material.index
    ]
    x = data.composition_fractions(row)
    dist = np.abs(material_x.to_numpy() - x.to_numpy()).sum(axis=1)
    i = int(np.argmin(dist))
    nearest = {
        "formula": str(per_material["formula"].iloc[i]),
        "tc_median": float(per_material["tc"].iloc[i]),
        "l1_distance": float(dist[i]),
        "within_0.05": int((dist < 0.05).sum()),
    }
    held = unique_m.loc[own, [data.FORMULA, data.TARGET]].reset_index(drop=True)
    return Query(formula, key, str(data.family(row).iloc[0]), x, X_train, y_train, held, nearest)


def _cache_name(formula: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", formula)


def _find_cached(q: Query) -> dict | None:
    """A cached prediction for this material (any spelling); its fingerprint must match."""
    fp = models.fingerprint(q.X_train, q.y_train, q.x, SEED, LEVELS)
    for path in sorted(CACHE.glob("*.json")):
        rec = json.loads(path.read_text())
        if rec["composition_key"] == q.key:
            if rec["fingerprint"] != fp:
                raise models.StaleCache(f"{path}: cached for different inputs")
            return rec
    return None


def predict(q: Query, live: bool, run_budget=None, predict_fn=models.tabpfn_predict) -> dict:
    rec = _find_cached(q)
    if rec is not None:
        return rec
    if not live:
        raise models.MissingCache(
            f"{q.formula}: no cached prediction. The demo calls the API only with --live "
            "(or TABPFN_LIVE=1) and a TABPFN_TOKEN; each new formula costs one request."
        )
    run_budget.charge(1, formula=q.formula)
    result = predict_fn(q.X_train, q.y_train, q.x, SEED, LEVELS)
    rec = {
        "formula": q.formula,
        "composition_key": q.key,
        "fingerprint": models.fingerprint(q.X_train, q.y_train, q.x, SEED, LEVELS),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_path": config.TABPFN_MODEL_PATH,
        "tabpfn_client": version("tabpfn-client"),
        "seed": SEED,
        "n_train": len(q.X_train),
        "n_held_out": len(q.held_out),
        "server_meta": result["meta"],
        "timings": result["timings"],
        "levels": [float(v) for v in LEVELS],
        "mean": float(result["mean"][0]),
        "quantiles": [float(v) for v in result["quantiles"][0]],
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{_cache_name(q.formula)}.json"
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")  # saved before any analysis
    return rec


def summarize(q: Query, rec: dict) -> dict:
    levels = np.asarray(rec["levels"])
    quant = np.asarray(rec["quantiles"])[None, :]

    def at(level: float) -> float:
        (i,) = np.flatnonzero(np.isclose(levels, level))
        return float(quant[0, i])

    held = q.held_out[data.TARGET]
    return {
        "formula": q.formula,
        "family": q.family,
        "mean": rec["mean"],
        "median": at(0.5),
        "q0.025": at(0.025),
        "q0.1": at(0.1),
        "q0.9": at(0.9),
        "q0.975": at(0.975),
        "p_above_77K": float(1 - metrics.cdf_at(quant, levels, 77.0)[0]),
        "in_data_rows": len(q.held_out),
        "recorded_tc_median": float(held.median()) if len(held) else np.nan,
        "recorded_tc_min": float(held.min()) if len(held) else np.nan,
        "recorded_tc_max": float(held.max()) if len(held) else np.nan,
        "nearest_formula": q.nearest["formula"],
        "nearest_tc_median": q.nearest["tc_median"],
        "nearest_l1": q.nearest["l1_distance"],
        "train_materials_within_0.05": q.nearest["within_0.05"],
    }


def format_probability(p: float) -> str:
    if p <= P_FLOOR + 1e-9:
        return f"< {P_FLOOR:.1%}"
    if p >= 1 - P_FLOOR - 1e-9:
        return f"> {1 - P_FLOOR:.1%}"
    return f"{p:.1%}"


def report(s: dict) -> str:
    lines = [
        f"{s['formula']}  ({s['family']})",
        f"  predicted Tc      {s['mean']:6.1f} K   (median {s['median']:.1f} K)",
        f"  80% interval      {s['q0.1']:6.1f} to {s['q0.9']:.1f} K",
        f"  95% interval      {s['q0.025']:6.1f} to {s['q0.975']:.1f} K",
        f"  P(Tc > 77 K)      {format_probability(s['p_above_77K'])}",
    ]
    if s["in_data_rows"]:
        lines.append(
            f"  in the data       {s['in_data_rows']} rows, recorded Tc median "
            f"{s['recorded_tc_median']:.1f} K (range {s['recorded_tc_min']:.1f} to "
            f"{s['recorded_tc_max']:.1f} K); held out of training for this prediction"
        )
    else:
        lines.append("  in the data       no")
    lines.append(
        f"  nearest training  {s['nearest_formula']} (L1 distance {s['nearest_l1']:.3f}, "
        f"Tc {s['nearest_tc_median']:.1f} K); {s['train_materials_within_0.05']} training "
        "materials within 0.05"
    )
    if s["nearest_l1"] >= 0.2:
        lines.append(
            "  warning           far from every training material: the interval may be too "
            "narrow (see README, 'Where TabPFN falls short')"
        )
    return "\n".join(lines)


def run(formulas: list[str], live: bool, predict_fn=models.tabpfn_predict) -> list[dict]:
    unique_m = data.load_unique_m()
    keys = data.composition_key(unique_m)
    queries = [build_query(f, unique_m, keys) for f in formulas]
    missing = [q for q in queries if _find_cached(q) is None]
    run_budget = None
    if missing and live:
        config.require_tabpfn_token()
        run_budget = budget.authorize(EXPERIMENT, len(missing), budget.MIN_TOKENS_PER_REQUEST)
    elif missing:
        names = ", ".join(q.formula for q in missing)
        raise models.MissingCache(
            f"no cached prediction for {names}. Rerun with --live (needs TABPFN_TOKEN; one "
            "request per formula). Cached formulas: `python -m demo.predict --list`."
        )
    return [summarize(q, predict(q, live, run_budget, predict_fn)) for q in queries]


def cached_formulas() -> list[str]:
    return [json.loads(p.read_text())["formula"] for p in sorted(CACHE.glob("*.json"))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("formulas", nargs="*", help='e.g. HgBa2Ca2Cu3O8 "La1.85Sr0.15CuO4"')
    parser.add_argument("--live", action="store_true", help="call the API for uncached formulas")
    parser.add_argument("--list", action="store_true", help="list the cached formulas")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)
    if args.list or not args.formulas:
        print("Cached formulas (no API key needed):\n  " + "\n  ".join(cached_formulas()))
        return 0
    try:
        results = run(args.formulas, config.live_requested(args.live))
    except (ValueError, models.MissingCache, budget.BudgetExceeded) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(results, indent=1))
    else:
        print("\n\n".join(report(s) for s in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
