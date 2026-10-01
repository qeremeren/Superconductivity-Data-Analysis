"""Train/test splits, defined once here and saved to data/splits/.

random:             the paper's protocol: 2/3 train / 1/3 test of rows, 25 repeats.
grouped:            the same proportions and repeats, but each scaled-composition
                    group goes wholly to one side, so no material is in both
                    train and test. The primary benchmark.
grouped_no_oxygen:  sensitivity check; groups by composition with oxygen dropped,
                    so all oxygen variants of a material (O6.9, O7, and formulas
                    with no oxygen amount) stay on one side.
leave_family_out:   3 folds; each holds out one family (cuprate, iron-based,
                    other) as the test set and trains on the other two.

`python -m src.splits` (or `make splits`) regenerates and saves them; a test
checks that the saved files equal a fresh regeneration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src import data

SPLITS_DIR = data.ROOT / "data" / "splits"
N_REPEATS = 25
TEST_FRACTION = 1 / 3
SEEDS = tuple(range(N_REPEATS))


def random_splits(n_rows: int, seeds=SEEDS) -> np.ndarray:
    """Boolean is_test masks, shape (len(seeds), n_rows)."""
    n_test = n_rows - round(n_rows * (1 - TEST_FRACTION))
    masks = np.zeros((len(seeds), n_rows), dtype=bool)
    for i, seed in enumerate(seeds):
        masks[i, np.random.default_rng(seed).permutation(n_rows)[:n_test]] = True
    return masks


def grouped_splits(groups, seeds=SEEDS) -> np.ndarray:
    """Boolean is_test masks, shape (len(seeds), len(groups)).

    Groups are shuffled per seed and assigned to test in that order, stopping at
    the group boundary closest to TEST_FRACTION of the rows.
    """
    labels, inverse, sizes = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    target = TEST_FRACTION * len(inverse)
    masks = np.zeros((len(seeds), len(inverse)), dtype=bool)
    for i, seed in enumerate(seeds):
        order = np.random.default_rng(seed).permutation(len(labels))
        n_test_groups = int(np.argmin(np.abs(np.cumsum(sizes[order]) - target))) + 1
        is_test_group = np.zeros(len(labels), dtype=bool)
        is_test_group[order[:n_test_groups]] = True
        masks[i] = is_test_group[inverse]
    return masks


FAMILY_ORDER = (data.CUPRATE, data.IRON_BASED, data.OTHER)


def leave_family_out(families) -> np.ndarray:
    """Boolean is_test masks, shape (3, n_rows): fold i tests on FAMILY_ORDER[i]."""
    families = np.asarray(families)
    return np.stack([families == f for f in FAMILY_ORDER])


def build_all() -> dict[str, np.ndarray]:
    um = data.load_unique_m()
    return {
        "random": random_splits(len(um)),
        "grouped": grouped_splits(data.composition_key(um)),
        "grouped_no_oxygen": grouped_splits(data.composition_key_without_oxygen(um)),
        "leave_family_out": leave_family_out(data.family(um)),
    }


def save(masks: np.ndarray, kind: str, splits_dir: Path = SPLITS_DIR) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    seeds = np.array(SEEDS[: len(masks)])
    np.savez_compressed(splits_dir / f"{kind}.npz", is_test=masks, seeds=seeds)


def seeds(kind: str, splits_dir: Path = SPLITS_DIR) -> np.ndarray:
    with np.load(splits_dir / f"{kind}.npz") as f:
        return f["seeds"]


def load(kind: str, splits_dir: Path = SPLITS_DIR) -> np.ndarray:
    with np.load(splits_dir / f"{kind}.npz") as f:
        return f["is_test"]


if __name__ == "__main__":
    for kind, masks in build_all().items():
        save(masks, kind)
        frac = masks.mean(axis=1)
        print(
            f"{kind}: {masks.shape[0]} splits, test fraction {frac.min():.4f}-{frac.max():.4f}, "
            f"written to {SPLITS_DIR / (kind + '.npz')}"
        )
