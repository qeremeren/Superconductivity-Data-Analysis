import json

import numpy as np
import pandas as pd
import pytest

from demo import predict
from src import budget, data, models


def test_parse_formula():
    assert predict.parse_formula("HgBa2Ca2Cu3O8") == {
        "Hg": 1, "Ba": 2, "Ca": 2, "Cu": 3, "O": 8,
    }  # fmt: skip
    assert predict.parse_formula("La1.85Sr0.15CuO4") == {
        "La": 1.85, "Sr": 0.15, "Cu": 1, "O": 4,
    }  # fmt: skip
    assert predict.parse_formula("(Ba0.6K0.4)Fe2As2") == pytest.approx(
        {"Ba": 0.6, "K": 0.4, "Fe": 2, "As": 2}
    )
    assert predict.parse_formula("Os2O") == {"Os": 2, "O": 1}
    for bad in ("", "UPt3", "Xx2", "Fe2(As", "Fe)2", "fe2as", "Fe-As"):
        with pytest.raises(ValueError):
            predict.parse_formula(bad)


def test_spellings_share_a_composition_key():
    key = lambda f: data.composition_key(predict.composition_row(f)).iloc[0]  # noqa: E731
    assert key("YBa2Cu3O7") == key("Y1Ba2Cu3O7") == key("Y0.5Ba1Cu1.5O3.5")


@pytest.fixture(scope="module")
def unique_m():
    if not data.is_downloaded():
        pytest.skip("data not downloaded")
    um = data.load_unique_m()
    return um, data.composition_key(um)


def test_known_material_is_held_out(unique_m):
    um, keys = unique_m
    q = predict.build_query("MgB2", um, keys)
    assert len(q.held_out) == (keys == q.key).sum() > 0
    assert len(q.X_train) == data.N_ROWS - len(q.held_out)
    assert q.nearest["l1_distance"] > 0  # the material itself is not among the neighbours
    assert q.family == data.OTHER


def test_live_path_with_a_fake_predictor(unique_m, tmp_path, monkeypatch):
    """Dry run of the billed path: the request is charged, then saved before analysis,
    and a second run reads it back from the cache without charging."""
    monkeypatch.setattr(predict, "CACHE", tmp_path / "predictions")
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(
        budget, "authorize",
        lambda exp, n, tok: budget.RunBudget(exp, n, tok, ledger=ledger),
    )  # fmt: skip
    monkeypatch.setattr(predict.config, "require_tabpfn_token", lambda: None)
    calls = []

    def fake(X_train, y_train, X_test, seed, levels):
        calls.append(len(X_train))
        q = np.linspace(0, 120, len(levels))[None, :]
        return {"mean": np.array([60.0]), "quantiles": q, "meta": {}, "timings": {}}

    with pytest.raises(models.MissingCache):
        predict.run(["LaH10"], live=False, predict_fn=fake)
    (s,) = predict.run(["LaH10"], live=True, predict_fn=fake)
    assert calls == [data.N_ROWS] and len(budget.read_ledger(ledger)) == 1
    assert s["in_data_rows"] == 0 and s["mean"] == 60.0
    assert s["median"] == pytest.approx(60.0, abs=1.0)
    assert 0.3 < s["p_above_77K"] < 0.4
    (again,) = predict.run(["La1H10"], live=False, predict_fn=fake)  # other spelling, cached
    assert again == s | {"formula": "La1H10"} and len(calls) == 1
    rec = json.loads(next((tmp_path / "predictions").glob("*.json")).read_text())
    assert rec["n_train"] == data.N_ROWS and len(rec["quantiles"]) == len(predict.LEVELS)


def test_committed_demo_cache_matches_inputs(unique_m):
    """Every committed demo prediction still fingerprints to the current data."""
    um, keys = unique_m
    formulas = predict.cached_formulas()
    if not formulas:
        pytest.skip("no committed demo predictions")
    for s in predict.run(formulas, live=False):
        assert np.isfinite(s["mean"]) and s["q0.025"] <= s["median"] <= s["q0.975"]
    assert isinstance(pd.read_csv(predict.OUT / "showcase.csv"), pd.DataFrame)
