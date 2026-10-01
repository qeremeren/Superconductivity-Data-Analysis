"""Download, verify and load the UCI "Superconductivty Data" set (id 464).

Run `python -m src.data` (or `make data`) to fetch the files into data/raw/.
Data: K. Hamidieh, UCI Machine Learning Repository, CC BY 4.0.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

UCI_PAGE = "https://archive.ics.uci.edu/dataset/464/superconductivty+data"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
CITATION = (
    "K. Hamidieh, A data-driven statistical model for predicting the critical "
    "temperature of a superconductor, Computational Materials Science 154 (2018) "
    "346-354, doi:10.1016/j.commatsci.2018.07.052"
)

# SHA-256 of each CSV inside the UCI zip, recorded from the download on 2026-09-28.
# We hash the CSVs rather than the zip because UCI may re-pack the archive.
SHA256 = {
    "train.csv": "4dfb6e3a1f6ffd969e5a5e42f093c4800d1e2a6c8b1e309f8fcd9f23d86952f3",
    "unique_m.csv": "b68ae6b55ea8581eff8b1ffba073a899db7e2d2f7f3b781bb0802f643f51e5f7",
}

TARGET = "critical_temp"
FORMULA = "material"
N_ROWS = 21_263
N_ENGINEERED_FEATURES = 81

# The element columns of unique_m.csv, in file (periodic-table) order.
ELEMENTS = tuple(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As "
    "Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd "
    "Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split()
)
N_ELEMENTS = len(ELEMENTS)

CUPRATE, IRON_BASED, OTHER = "cuprate", "iron-based", "other"
KEY_DECIMALS = 6


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_downloaded(raw_dir: Path = RAW_DIR) -> bool:
    """True if every expected CSV exists in raw_dir with the pinned checksum."""
    return all(
        (raw_dir / name).is_file() and sha256_of(raw_dir / name) == digest
        for name, digest in SHA256.items()
    )


def download(raw_dir: Path = RAW_DIR, force: bool = False) -> None:
    """Fetch the UCI zip, verify each CSV against SHA256, and extract to raw_dir.

    Every file is verified in memory before anything is written, so a changed
    upstream file raises instead of leaving a half-updated data/raw/.
    """
    if not force and is_downloaded(raw_dir):
        print(f"Data already present and verified in {raw_dir}")
        return

    print(f"Downloading {UCI_ZIP_URL}")
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=120) as resp:
        payload = resp.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        contents = {name: zf.read(name) for name in SHA256}

    for name, blob in contents.items():
        digest = hashlib.sha256(blob).hexdigest()
        if digest != SHA256[name]:
            raise RuntimeError(
                f"Checksum mismatch for {name}: expected {SHA256[name]}, got {digest}. "
                "The upstream UCI file has changed; do not update the pinned hash "
                "without re-checking the data."
            )

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in contents.items():
        (raw_dir / name).write_bytes(blob)

    source = {
        "dataset": "Superconductivty Data (UCI id 464)",
        "page": UCI_PAGE,
        "url": UCI_ZIP_URL,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "license": LICENSE,
        "citation": CITATION,
        "sha256": SHA256,
    }
    (raw_dir / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"Wrote {', '.join(SHA256)} to {raw_dir} (checksums verified)")


def _read_csv(name: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    return pd.read_csv(path)


def load_train(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """The 81 engineered features plus critical_temp, one row per UCI record."""
    df = _read_csv("train.csv", raw_dir)
    assert df.shape == (N_ROWS, N_ENGINEERED_FEATURES + 1), df.shape
    assert df.columns[-1] == TARGET
    return df


def load_unique_m(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Element composition (86 columns) plus critical_temp and the formula string."""
    df = _read_csv("unique_m.csv", raw_dir)
    assert df.shape == (N_ROWS, N_ELEMENTS + 2), df.shape
    assert tuple(df.columns[:N_ELEMENTS]) == ELEMENTS
    assert list(df.columns[-2:]) == [TARGET, FORMULA]
    return df


def composition_fractions(unique_m: pd.DataFrame) -> pd.DataFrame:
    """Element fractions summing to 1 per row, so scaled formulas coincide."""
    counts = unique_m[list(ELEMENTS)]
    return counts.div(counts.sum(axis=1), axis=0)


def composition_key(unique_m: pd.DataFrame) -> pd.Series:
    """Scaled-composition key such as "O:0.538462 Cu:0.230769 Y:0.076923 Ba:0.153846".

    Rows with the same key are one material for grouping and deduplication:
    YBa2Cu3O7 and Y0.5Ba1Cu1.5O3.5 share a key, whatever the formula spelling.
    """
    fractions = composition_fractions(unique_m).round(KEY_DECIMALS).to_numpy()
    names = np.array(ELEMENTS)

    def key(row: np.ndarray) -> str:
        present = row > 0
        pairs = zip(names[present], row[present], strict=True)
        return " ".join(f"{e}:{v:.{KEY_DECIMALS}f}" for e, v in pairs)

    return pd.Series([key(row) for row in fractions], index=unique_m.index, name="composition_key")


def family(unique_m: pd.DataFrame) -> pd.Series:
    """Cuprate: Cu > 0 and O > 0. Iron-based: Fe > 0 and (As > 0 or Se > 0).
    Other: everything else. Cuprate wins if both match (none do in this data)."""
    cuprate = (unique_m["Cu"] > 0) & (unique_m["O"] > 0)
    iron = (unique_m["Fe"] > 0) & ((unique_m["As"] > 0) | (unique_m["Se"] > 0))
    labels = np.select([cuprate, iron], [CUPRATE, IRON_BASED], default=OTHER)
    return pd.Series(labels, index=unique_m.index, name="family")


FEATURE_SETS = ("engineered", "composition")


def feature_sets(train: pd.DataFrame, unique_m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """engineered: the paper's 81 features. composition: the 86 element fractions
    (summing to 1 per row), so scaled formulas get identical inputs."""
    return {
        "engineered": train.drop(columns=[TARGET]),
        "composition": composition_fractions(unique_m),
    }


def composition_key_without_oxygen(unique_m: pd.DataFrame) -> pd.Series:
    """Scaled composition after dropping oxygen: all oxygen variants of a material
    (YBa2Cu3O6.9, YBa2Cu3O7, and YBa2Cu3O with no amount given) share one key."""
    return composition_key(unique_m[list(ELEMENTS)].assign(O=0.0))


def oxygen_amount_missing(unique_m: pd.DataFrame) -> pd.Series:
    """Formula has an "O" with no amount (e.g. Y1Ba2Cu3O); unique_m.csv encodes it as O = 1.
    "O" followed by a lowercase letter is another element (Os)."""
    return unique_m[FORMULA].str.contains(r"O(?![a-z\d.])", regex=True).rename("oxygen_missing")


def suspect_reasons(unique_m: pd.DataFrame) -> pd.Series:
    """Rule-based reasons a row is doubtful as a discovery target ("" if none).

    Oxides with Ba or Sr but no Cu above 40 K look like cuprates whose formula lost
    its Cu (e.g. Y1Ba23O for YBa2Cu3O). Non-cuprates above 77 K are rare enough to
    review one by one. Used to exclude materials from the discovery pools only.
    """
    tc, fam = unique_m[TARGET], family(unique_m)
    oxide = (fam == OTHER) & (unique_m["O"] > 0) & ((unique_m["Ba"] > 0) | (unique_m["Sr"] > 0))
    masks = {
        "cuprate-like oxide without Cu, Tc > 40 K": oxide & (unique_m["Cu"] == 0) & (tc > 40),
        "non-cuprate above 77 K": (fam != CUPRATE) & (tc > 77),
    }
    reasons = pd.Series("", index=unique_m.index, name="suspect")
    for reason, mask in masks.items():
        reasons[mask] = np.where(reasons[mask] == "", reason, reasons[mask] + "; " + reason)
    return reasons


if __name__ == "__main__":
    download()
