"""Shared inputs and paths for the Phase 2 benchmark scripts."""

from __future__ import annotations

import os
import platform
import socket
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import pandas as pd

from src import config, data, splits

SPLIT_KINDS = ("random", "grouped", "grouped_no_oxygen", "leave_family_out")
OUT = config.RESULTS_DIR / "02_benchmark"


@dataclass(frozen=True)
class Inputs:
    features: dict[str, pd.DataFrame]
    y: np.ndarray
    family: np.ndarray
    oxygen_missing: np.ndarray
    composition_key: np.ndarray
    composition_key_without_oxygen: np.ndarray


def load_inputs() -> Inputs:
    train, um = data.load_train(), data.load_unique_m()
    return Inputs(
        features=data.feature_sets(train, um),
        y=train[data.TARGET].to_numpy(float),
        family=data.family(um).to_numpy(),
        oxygen_missing=data.oxygen_amount_missing(um).to_numpy(),
        composition_key=data.composition_key(um).to_numpy(),
        composition_key_without_oxygen=data.composition_key_without_oxygen(um).to_numpy(),
    )


def inner_groups(kind: str, inputs: Inputs) -> np.ndarray | None:
    """Grouping for the inner tuning split, mirroring the outer split type."""
    if kind == "random":
        return None
    if kind == "grouped_no_oxygen":
        return inputs.composition_key_without_oxygen
    return inputs.composition_key  # grouped, leave_family_out


def iter_splits(kinds=SPLIT_KINDS):
    """(kind, index, seed, is_test) for every saved split."""
    for kind in kinds:
        for i, (seed, is_test) in enumerate(
            zip(splits.seeds(kind), splits.load(kind), strict=True)
        ):
            yield kind, i, int(seed), is_test


def predictions_path(model: str, kind: str, features: str):
    return OUT / "predictions" / model / kind / f"{features}.parquet"


def write_predictions(rows: list[pd.DataFrame], model: str, kind: str, features: str) -> None:
    path = predictions_path(model, kind, features)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(path, compression="zstd", index=False)


def prediction_frame(split: int, test_rows: np.ndarray, pred) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": np.int16(split),
            "row": np.asarray(test_rows, np.int32),
            "pred": np.asarray(pred, np.float32),
        }
    )


LC_SPLITS = (0, 1, 2, 3, 4)  # grouped splits used for learning curves
LC_SIZES = (100, 300, 1000, 3000, 10000)  # training rows; the full size is Phase 2


def learning_curve_rows(is_test: np.ndarray, split: int, n: int) -> np.ndarray:
    """The first n of a fixed, seeded permutation of the split's training rows, so the
    subsets are nested across sizes and identical for every model."""
    train_rows = np.flatnonzero(~is_test)
    perm = np.random.default_rng(10_000 + split).permutation(len(train_rows))
    return np.sort(train_rows[perm[:n]])


def machine_info() -> dict:
    """Which machine ran a job, recorded with every local result."""
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "xgboost": version("xgboost"),
        "numpy": version("numpy"),
    }
