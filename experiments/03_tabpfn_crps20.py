"""TabPFN CRPS on the 20-level midpoint grid, for a like-for-like comparison with the
XGBoost uncertainty baselines (which are scored on the same 20 levels).

Needs the full 107-level grids, which exist only in the local raw cache of the machine
that made the Phase 2 requests (results/cache/tabpfn_raw/, gitignored). Each grid is
checked against the committed request record's fingerprint before use. The output,
results/03_uncertainty/tabpfn_crps20.parquet, is committed, so reproducing never needs
the raw grids. Covers the grouped split and the leave-family-out folds.

Run from the repo root on the machine with the raw cache:
`uv run python -m experiments.03_tabpfn_crps20`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import benchmark, data, metrics, models

KINDS = ("grouped", "leave_family_out")
OUT = benchmark.config.RESULTS_DIR / "03_uncertainty" / "tabpfn_crps20.parquet"
SUMMARIES = benchmark.OUT / "tabpfn"
RAW = models.RAW_CACHE / "tabpfn"


def main():
    frames = []
    for kind in KINDS:
        for fs in data.FEATURE_SETS:
            for summary_path in sorted((SUMMARIES / kind / fs).glob("split_*.parquet")):
                split = int(summary_path.stem.split("_")[1])
                record = json.loads(summary_path.with_suffix(".json").read_text())
                raw_path = RAW / kind / fs / summary_path.with_suffix(".npz").name
                if not raw_path.is_file():
                    raise SystemExit(f"{raw_path} missing: run this where the requests were made.")
                with np.load(raw_path) as raw:
                    if str(raw["fingerprint"]) != record["fingerprint"]:
                        raise SystemExit(f"{raw_path} does not match its committed record")
                    q, levels = raw["quantiles"].astype(float), raw["levels"]
                y = pd.read_parquet(summary_path, columns=["row", "y"])
                crps20 = metrics.crps(q, levels, y["y"].to_numpy(), metrics.MIDPOINT_20)
                frames.append(
                    pd.DataFrame(
                        {
                            "kind": kind,
                            "features": fs,
                            "split": np.int16(split),
                            "row": y["row"].astype(np.int32),
                            "crps20": crps20.astype(np.float32),
                        }
                    )
                )
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, compression="zstd", index=False)
    print(f"Wrote {len(out):,} rows from {len(frames)} jobs to {OUT}")


if __name__ == "__main__":
    main()
