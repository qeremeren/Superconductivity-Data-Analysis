import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src import data

needs_data = pytest.mark.skipif(
    not data.is_downloaded(), reason="data/raw/ missing or unverified; run `make data`"
)


@needs_data
def test_pinned_checksums():
    for name, digest in data.SHA256.items():
        assert data.sha256_of(data.RAW_DIR / name) == digest


@needs_data
def test_shapes_and_columns():
    train = data.load_train()
    unique_m = data.load_unique_m()
    assert train.shape == (21_263, 82)
    assert unique_m.shape == (21_263, 88)
    assert train.columns[-1] == "critical_temp"
    assert list(unique_m.columns[-2:]) == ["critical_temp", "material"]
    assert unique_m.columns[0] == "H" and unique_m.columns[-3] == "Rn"


@needs_data
def test_no_missing_values():
    assert not data.load_train().isna().any().any()
    assert not data.load_unique_m().isna().any().any()


@needs_data
def test_files_are_row_aligned():
    # Same target in the same row is necessary but not sufficient, so also check
    # that train.csv's element count equals the nonzero entries in unique_m.csv.
    train = data.load_train()
    unique_m = data.load_unique_m()
    np.testing.assert_array_equal(train["critical_temp"], unique_m["critical_temp"])
    elements = unique_m.columns[: data.N_ELEMENTS]
    n_nonzero = (unique_m[elements] > 0).sum(axis=1)
    np.testing.assert_array_equal(train["number_of_elements"], n_nonzero)


def _compositions(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(0.0, index=range(len(rows)), columns=list(data.ELEMENTS))
    for i, row in enumerate(rows):
        for element, amount in row.items():
            df.loc[i, element] = amount
    return df


def test_composition_key_ignores_scaling_but_not_stoichiometry():
    df = _compositions(
        [
            {"Y": 1, "Ba": 2, "Cu": 3, "O": 7},
            {"Y": 0.5, "Ba": 1, "Cu": 1.5, "O": 3.5},
            {"Y": 1, "Ba": 2, "Cu": 3, "O": 6.9},
        ]
    )
    key = data.composition_key(df)
    assert key[0] == key[1]
    assert key[0] != key[2]
    assert key[0] == "O:0.538462 Cu:0.230769 Y:0.076923 Ba:0.153846"


def test_family_rules():
    df = _compositions(
        [
            {"La": 2, "Cu": 1, "O": 4},
            {"Ba": 1, "Fe": 2, "As": 2},
            {"Fe": 1, "Se": 1},
            {"Nb": 3, "Cu": 1},
            {"Fe": 1, "O": 1},
            {"Mg": 1, "B": 2},
        ]
    )
    expected = ["cuprate", "iron-based", "iron-based", "other", "other", "other"]
    assert list(data.family(df)) == expected


@needs_data
def test_family_definitions_do_not_overlap_in_data():
    um = data.load_unique_m()
    cuprate = (um["Cu"] > 0) & (um["O"] > 0)
    iron = (um["Fe"] > 0) & ((um["As"] > 0) | (um["Se"] > 0))
    assert not (cuprate & iron).any()


def _fake_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return buf.getvalue()


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_rejects_changed_upstream_file(tmp_path, monkeypatch):
    payload = _fake_zip({"train.csv": b"tampered", "unique_m.csv": b"tampered"})
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        data.download(raw_dir=tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


def test_download_writes_verified_files(tmp_path, monkeypatch):
    files = {"train.csv": b"a,b\n1,2\n", "unique_m.csv": b"c\n3\n"}
    monkeypatch.setattr(
        data, "SHA256", {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    )
    payload = _fake_zip(files)
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))
    raw = tmp_path / "raw"
    data.download(raw_dir=raw)
    assert data.is_downloaded(raw)
    assert (raw / "SOURCE.json").is_file()


def test_loader_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="make data"):
        data.load_train(raw_dir=tmp_path)
