"""Discovery pools for the Phase 4 loop (the loop itself is added in Phase 4).

A pool has one row per material (scaled composition) with Tc = median over the
material's rows. Materials with any row flagged by data.suspect_reasons are left
out of every pool (the benchmarks keep them) and listed, with reasons, in
data/discovery_exclusions.csv; `python -m src.discovery` regenerates that file.

Scenarios: "main" = all remaining materials; "hard" = non-cuprates only. The
target in both is the top TOP_FRACTION of the pool's Tc.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import data

EXCLUSIONS = data.ROOT / "data" / "discovery_exclusions.csv"
TOP_FRACTION = 0.01
SCENARIOS = ("main", "hard")


def materials(unique_m: pd.DataFrame) -> pd.DataFrame:
    """Every material, indexed by composition key, with its suspect reasons ("" if none)."""
    g = pd.DataFrame(
        {
            "key": data.composition_key(unique_m),
            "family": data.family(unique_m),
            "tc": unique_m[data.TARGET],
            "formula": unique_m[data.FORMULA],
            "suspect": data.suspect_reasons(unique_m),
        }
    )
    grp = g.groupby("key")
    return pd.DataFrame(
        {
            "family": grp["family"].first(),
            "tc_median": grp["tc"].median(),
            "n_rows": grp.size(),
            "formulas": grp["formula"].agg(lambda s: "; ".join(pd.unique(s)[:3])),
            "suspect": grp["suspect"].agg(lambda s: "; ".join(sorted({r for r in s if r}))),
        }
    ).rename_axis("composition_key")


def exclusions(unique_m: pd.DataFrame) -> pd.DataFrame:
    m = materials(unique_m)
    return m[m.suspect != ""].sort_values("tc_median", ascending=False)


def build_pool(unique_m: pd.DataFrame, scenario: str = "main") -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    m = materials(unique_m)
    pool = m[m.suspect == ""]
    if scenario == "hard":
        pool = pool[pool.family != data.CUPRATE]
    return pool.drop(columns="suspect")


def top_threshold(tc, fraction: float = TOP_FRACTION) -> float:
    """Tc of the ceil(fraction * N)-th highest material; ties at it also count as targets."""
    tc = np.sort(np.asarray(tc, dtype=float))[::-1]
    return float(tc[math.ceil(fraction * len(tc)) - 1])


def write_exclusions(path: Path = EXCLUSIONS) -> pd.DataFrame:
    table = exclusions(data.load_unique_m())
    table.to_csv(path, float_format="%.6g")
    return table


if __name__ == "__main__":
    table = write_exclusions()
    print(f"Wrote {len(table)} excluded materials to {EXCLUSIONS}")
