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
N_ELEMENTS = 86


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
    assert list(df.columns[-2:]) == [TARGET, FORMULA]
    return df


if __name__ == "__main__":
    download()
