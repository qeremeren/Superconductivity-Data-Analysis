"""Discovery pools and the Phase 4 discovery loop.

A pool has one row per material (scaled composition) with Tc = median over the
material's rows. Materials with any row flagged by data.suspect_reasons are left
out of every pool (the benchmarks keep them) and listed, with reasons, in
data/discovery_exclusions.csv; `python -m src.discovery` regenerates that file.

Scenarios: "main" = all remaining materials; "hard" = non-cuprates only. The
target in both is the top TOP_FRACTION of the pool's Tc.

The loop (run_loop) is pure and deterministic given a predictor: it starts from
N_INITIAL random non-target materials, and each round fits on the labeled set,
scores every unlabeled material, and reveals the top BATCH. Acquisitions:
  ei             expected improvement over y* = min(best Tc so far, target threshold)
  q90            the predictive 0.9 quantile
  greedy_tabpfn  TabPFN's predictive mean
  greedy_xgb     XGBoost (published settings) prediction
  random         uniform over unlabeled materials
TabPFN scores come from the tail-dense quantile grid TAIL_LEVELS (0.01 steps up to the
0.95 quantile, 0.001 steps above), so tail quantities resolve to 0.1%.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from src import data, metrics

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


# --- the discovery loop ------------------------------------------------------------------

N_INITIAL = 50
BATCH = 10
ROUNDS = 20
PILOT_SEEDS = (100, 101, 102)
MAIN_SEEDS = tuple(range(10))
# Post hoc (Phase 4 evaluation, chosen after seeing the per-seed results): a run "stalled" if
# it found at most this many targets in 200 experiments (random search expects ~2).
# stalls.csv reports the gap between stalled and other runs, so the cut-off's role is visible.
STALL_MAX = 5
SCENARIO_IDS = {"main": 1, "hard": 2}
TABPFN_ACQUISITIONS = ("ei", "q90", "greedy_tabpfn")
FREE_ACQUISITIONS = ("greedy_xgb", "random")
P_BINS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def _tail_grid() -> tuple[np.ndarray, np.ndarray]:
    """Levels and midpoint-rule weights: 95 midpoints of width 0.01 below the 0.95
    quantile, 50 of width 0.001 above, plus 0.9 itself (weight 0) for q90."""
    body = np.round((np.arange(95) + 0.5) / 100, 4)
    tail = np.round(0.95 + (np.arange(50) + 0.5) / 1000, 4)
    levels = np.sort(np.concatenate([body, tail, [0.9]]))
    weights = np.where(np.isin(levels, body), 0.01, np.where(np.isin(levels, tail), 0.001, 0.0))
    return levels, weights


TAIL_LEVELS, TAIL_WEIGHTS = _tail_grid()


def expected_improvement(quantiles: np.ndarray, weights: np.ndarray, y_star: float) -> np.ndarray:
    """E[max(Y - y*, 0)] per row by the midpoint rule over the quantile grid."""
    return (np.maximum(np.asarray(quantiles) - y_star, 0.0) * weights).sum(axis=1)


def tabpfn_scores(mean, quantiles, levels, weights, y_star, threshold) -> pd.DataFrame:
    q = np.asarray(quantiles, float)
    (i90,) = np.flatnonzero(np.isclose(levels, 0.9))
    return pd.DataFrame(
        {
            "mean": np.asarray(mean, float),
            "ei": expected_improvement(q, weights, y_star),
            "q90": q[:, i90],
            "p_top1": 1 - metrics.cdf_at(q, levels, threshold),
            "p_77K": 1 - metrics.cdf_at(q, levels, 77.0),
        }
    )


@dataclass(frozen=True)
class Scenario:
    name: str
    keys: np.ndarray  # composition keys, one per pool material
    X: pd.DataFrame  # composition fractions
    tc: np.ndarray  # median Tc
    family: np.ndarray
    threshold: float

    @property
    def is_target(self) -> np.ndarray:
        return self.tc >= self.threshold


def load_scenario(name: str, unique_m: pd.DataFrame | None = None) -> Scenario:
    um = data.load_unique_m() if unique_m is None else unique_m
    pool = build_pool(um, name)
    key = data.composition_key(um)
    first = pd.Series(np.arange(len(um))).groupby(key.to_numpy()).first()
    fractions = data.composition_fractions(um).iloc[first.loc[pool.index].to_numpy()]
    return Scenario(
        name=name,
        keys=pool.index.to_numpy(),
        X=fractions.reset_index(drop=True),
        tc=pool.tc_median.to_numpy(float),
        family=pool.family.to_numpy(),
        threshold=top_threshold(pool.tc_median),
    )


def initial_set(scn: Scenario, seed: int) -> np.ndarray:
    """N_INITIAL random non-target materials (pool indices), the same for every
    acquisition with this seed and scenario."""
    rng = np.random.default_rng([seed, SCENARIO_IDS[scn.name]])
    return np.sort(rng.choice(np.flatnonzero(~scn.is_target), N_INITIAL, replace=False))


def select_batch(score, mean, candidates, batch, rng) -> np.ndarray:
    """The top `batch` candidates by score; ties broken by mean, then at random."""
    order = np.lexsort((rng.random(len(candidates)), -np.asarray(mean), -np.asarray(score)))
    return candidates[order[:batch]]


def _bins(p, hit) -> list[dict]:
    idx = np.clip(np.digitize(p, P_BINS) - 1, 0, len(P_BINS) - 2)
    out = []
    for b in range(len(P_BINS) - 1):
        m = idx == b
        out.append(
            {
                "lo": P_BINS[b],
                "hi": P_BINS[b + 1],
                "n": int(m.sum()),
                "sum_p": float(p[m].sum()),
                "hits": int(hit[m].sum()),
            }
        )
    return out


def labeled_fingerprint(keys) -> str:
    return hashlib.sha256("|".join(sorted(map(str, keys))).encode()).hexdigest()[:16]


def run_loop(
    scn, acquisition, seed, predict=None, record=None, on_round=None, rounds=ROUNDS, batch=BATCH
):
    """Run (or resume) one discovery run. predict(X_train, y_train, X_pool, seed) returns
    {"mean", "quantiles" (or None), "meta", "seconds"}; random needs none. `record` is a
    previous run record to resume from. on_round(record, round, scores_frame, raw) is
    called after every round so the caller can persist it. Returns the record."""
    if record is None:
        init = initial_set(scn, seed)
        record = {
            "scenario": scn.name,
            "acquisition": acquisition,
            "seed": int(seed),
            "n_initial": N_INITIAL,
            "batch": batch,
            "rounds_planned": rounds,
            "threshold_K": scn.threshold,
            "pool_size": int(len(scn.keys)),
            "targets_in_pool": int(scn.is_target.sum()),
            "grid_levels": len(TAIL_LEVELS) if acquisition in TABPFN_ACQUISITIONS else None,
            "initial": [str(k) for k in scn.keys[init]],
            "rounds": [],
            "status": "running",
        }
    index = {k: i for i, k in enumerate(scn.keys)}
    labeled = [index[k] for k in record["initial"]]
    for r in record["rounds"]:
        labeled += [index[s["key"]] for s in r["selected"]]
    found = sum(int(s["is_target"]) for r in record["rounds"] for s in r["selected"])
    for rnd in range(len(record["rounds"]) + 1, rounds + 1):
        lab = np.array(labeled)
        unl = np.setdiff1d(np.arange(len(scn.keys)), lab)
        best = float(scn.tc[lab].max())
        y_star = min(best, scn.threshold)
        rng = np.random.default_rng([seed, SCENARIO_IDS[scn.name], rnd, 7])
        raw, info = None, {}
        if acquisition == "random":
            sc = pd.DataFrame({"mean": np.full(len(unl), np.nan), "score": rng.random(len(unl))})
        else:
            out = predict(scn.X.iloc[lab], scn.tc[lab], scn.X.iloc[unl], seed)
            info = {"seconds": out.get("seconds"), "model": out.get("meta", {}).get("model_path")}
            if out.get("quantiles") is None:
                sc = pd.DataFrame({"mean": np.asarray(out["mean"], float)})
            else:
                sc = tabpfn_scores(
                    out["mean"], out["quantiles"], TAIL_LEVELS, TAIL_WEIGHTS, y_star, scn.threshold
                )
                raw = out
            key_col = {"ei": "ei", "q90": "q90"}.get(acquisition, "mean")
            sc["score"] = sc[key_col]
        picked = select_batch(
            sc["score"].to_numpy(), sc["mean"].fillna(0).to_numpy(), unl, batch, rng
        )
        pos = np.searchsorted(unl, picked)
        selected = []
        for i, p in zip(picked, pos, strict=True):
            row = {
                "key": str(scn.keys[i]),
                "tc": float(scn.tc[i]),
                "is_target": bool(scn.is_target[i]),
                "family": str(scn.family[i]),
            }
            for c in ("mean", "ei", "q90", "p_top1", "p_77K"):
                if c in sc and not np.isnan(sc[c].iloc[p]):
                    row[c] = float(sc[c].iloc[p])
            selected.append(row)
        found += sum(s["is_target"] for s in selected)
        pool_stats = {
            "n_unlabeled": int(len(unl)),
            "targets_remaining": int(scn.is_target[unl].sum()),
        }
        if "p_top1" in sc:
            hit = scn.is_target[unl]
            pool_stats |= {
                "sum_p_top1": float(sc.p_top1.sum()),
                "sum_p_77K": float(sc.p_77K.sum()),
                "max_ei": float(sc.ei.max()),
                "p_top1_bins": _bins(sc.p_top1.to_numpy(), hit),
            }
        record["rounds"].append(
            {
                "round": rnd,
                "n_labeled": int(len(lab)),
                "best_so_far_K": best,
                "y_star_K": y_star,
                "labeled_fingerprint": labeled_fingerprint(scn.keys[lab]),
                "selected": selected,
                "found_cumulative": int(found),
                "pool": pool_stats,
                "request": info,
            }
        )
        labeled += list(picked)
        if on_round is not None:
            on_round(record, rnd, sc.assign(pool_index=unl), raw)
    record["status"] = "complete"
    return record


# --- gate checks (independent of which method wins) -------------------------------------


def check_run(record: dict, scn: Scenario, rounds: int = ROUNDS) -> list[str]:
    """Every problem with one run record: missing or misnumbered rounds, wrong batch sizes,
    selections that were not unlabeled, a start set with targets, Tc not matching the data."""
    problems = []
    name = f"{record.get('scenario')}/{record.get('acquisition')}/seed {record.get('seed')}"
    index = {k: i for i, k in enumerate(scn.keys)}
    if record.get("status") != "complete":
        problems.append(f"{name}: status {record.get('status')}")
    nums = [r["round"] for r in record.get("rounds", [])]
    if nums != list(range(1, rounds + 1)):
        problems.append(f"{name}: rounds {nums[:3]}...{nums[-3:]} (expected 1..{rounds})")
    init = record.get("initial", [])
    if len(set(init)) != N_INITIAL or any(k not in index for k in init):
        problems.append(f"{name}: start set is not {N_INITIAL} distinct pool materials")
    elif any(scn.is_target[index[k]] for k in init):
        problems.append(f"{name}: start set contains a target")
    seen = set(init)
    for r in record.get("rounds", []):
        keys = [s["key"] for s in r["selected"]]
        if len(keys) != record.get("batch", BATCH) or len(set(keys)) != len(keys):
            problems.append(f"{name} round {r['round']}: batch size or duplicates")
        if any(k in seen for k in keys):
            problems.append(f"{name} round {r['round']}: selected an already labeled material")
        if any(k not in index for k in keys):
            problems.append(f"{name} round {r['round']}: selected a material outside the pool")
            continue
        for s in r["selected"]:
            i = index[s["key"]]
            if not np.isclose(s["tc"], scn.tc[i]) or s["is_target"] != bool(scn.is_target[i]):
                problems.append(f"{name} round {r['round']}: Tc/target label mismatch")
        seen |= set(keys)
    return problems


def random_expectation(scn: Scenario, experiments: int) -> dict:
    """Exact hypergeometric law of targets found by uniform random picks after a
    target-free start set."""
    n_unlabeled, k = len(scn.keys) - N_INITIAL, int(scn.is_target.sum())
    dist = hypergeom(n_unlabeled, k, experiments)
    return {
        "mean": float(dist.mean()),
        "sd": float(dist.std()),
        "p_at_least_one": float(1 - dist.pmf(0)),
    }


def simulate_random(scn: Scenario, replicates: int, rounds=ROUNDS, batch=BATCH) -> np.ndarray:
    """Targets found by the random acquisition, through run_loop, for many seeds."""
    return np.array(
        [
            run_loop(scn, "random", 10_000 + i, rounds=rounds, batch=batch)["rounds"][-1][
                "found_cumulative"
            ]
            for i in range(replicates)
        ]
    )


def pilot_decision(found: dict[str, list[int]]) -> dict:
    """Pre-registered rule: the acquisition with more targets found by the last round,
    summed over all pilot runs, wins; within max(1, 10% of the larger total), keep both."""
    totals = {a: int(sum(v)) for a, v in found.items()}
    (a, ta), (b, tb) = sorted(totals.items(), key=lambda kv: -kv[1])[:2]
    tie = abs(ta - tb) <= max(1, 0.1 * max(ta, tb))
    return {
        "totals": totals,
        "tie": bool(tie),
        "chosen": sorted([a, b]) if tie else [a],
        "rule": "more targets found by the last round summed over pilot runs; "
        "within max(1, 10% of the larger total) keeps both",
    }


def fake_predict(X_train, y_train, X_pool, seed):
    """Offline stand-in for TabPFN, for dry runs and tests only: a 1-NN mean on
    composition with a Gaussian spread on the tail-dense grid. Never used for results."""
    from scipy.stats import norm

    Xt, Xp = np.asarray(X_train, float), np.asarray(X_pool, float)
    d = np.abs(Xp[:, None, :] - Xt[None, :, :]).sum(-1) if len(Xp) * len(Xt) < 4e6 else None
    if d is None:
        nn = np.array([np.abs(Xt - x).sum(1).argmin() for x in Xp])
    else:
        nn = d.argmin(1)
    mean = np.asarray(y_train, float)[nn]
    q = mean[:, None] + 15.0 * norm.ppf(TAIL_LEVELS)[None, :]
    return {"mean": mean, "quantiles": q, "meta": {"model_path": "fake"}, "seconds": 0.0}


if __name__ == "__main__":
    table = write_exclusions()
    print(f"Wrote {len(table)} excluded materials to {EXCLUSIONS}")
