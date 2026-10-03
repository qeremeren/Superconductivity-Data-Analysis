import pandas as pd
import pytest

from src import data, discovery

needs_data = pytest.mark.skipif(
    not data.is_downloaded(), reason="data/raw/ missing or unverified; run `make data`"
)


def test_top_threshold_counts_ties():
    tc = [150, 140, 130, 130, 120] + [10] * 295  # 300 materials -> top 3
    assert discovery.top_threshold(tc) == 130.0


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        discovery.build_pool(pd.DataFrame(), scenario="easy")


@needs_data
def test_committed_exclusions_match_the_rule():
    saved = pd.read_csv(discovery.EXCLUSIONS, index_col="composition_key")
    fresh = discovery.exclusions(data.load_unique_m())
    assert list(saved.index) == list(fresh.index)
    assert list(saved.suspect) == list(fresh.suspect)


@needs_data
def test_pools_drop_exclusions_and_hard_pool_has_no_cuprates():
    um = data.load_unique_m()
    excluded = set(discovery.exclusions(um).index)
    main, hard = discovery.build_pool(um, "main"), discovery.build_pool(um, "hard")
    assert not excluded & set(main.index)
    assert not excluded & set(hard.index)
    assert (hard.family != data.CUPRATE).all()
    assert set(hard.index) <= set(main.index)
    assert main.index.is_unique


# --- the discovery loop (synthetic pool, no data or API needed) ---------------------------

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402


def _toy_scenario(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.dirichlet(np.ones(5), size=n), columns=list("abcde"))
    tc = 100 * X["a"].to_numpy() + rng.normal(0, 3, n)
    return discovery.Scenario(
        name="main",
        keys=np.array([f"m{i:04d}" for i in range(n)]),
        X=X,
        tc=tc,
        family=np.array(["other"] * n),
        threshold=discovery.top_threshold(tc, 0.02),
    )


def test_tail_grid_weights_and_levels():
    lv, w = discovery.TAIL_LEVELS, discovery.TAIL_WEIGHTS
    assert len(lv) == 146 and np.isclose(w.sum(), 1.0)
    assert np.isclose(lv, 0.9).sum() == 1 and w[np.isclose(lv, 0.9)][0] == 0
    assert np.isclose(np.diff(lv[lv > 0.95]), 0.001).all()


def test_expected_improvement_tail_grid_matches_999_grid_and_closed_form():
    # Both grids end at the 0.9995 quantile, so beyond ~2 sd both underestimate EI by the
    # same amount; the tail-dense grid must match the 999-level grid everywhere.
    mu, sd = 50.0, 10.0
    u999 = (np.arange(999) + 0.5) / 999
    for y_star in (40.0, 60.0, 70.0, 75.0, 80.0):
        tail = discovery.expected_improvement(
            stats.norm.ppf(discovery.TAIL_LEVELS, mu, sd)[None], discovery.TAIL_WEIGHTS, y_star
        )[0]
        uni = discovery.expected_improvement(
            stats.norm.ppf(u999, mu, sd)[None], np.full(999, 1 / 999), y_star
        )[0]
        assert tail == pytest.approx(uni, rel=0.01)
        z = (mu - y_star) / sd
        if z >= -2:
            exact = sd * stats.norm.pdf(z) + (mu - y_star) * stats.norm.cdf(z)
            assert tail == pytest.approx(exact, rel=0.015)


def test_batches_come_only_from_candidates():
    rng = np.random.default_rng(0)
    cand = np.array([3, 7, 9, 11, 20])
    picked = discovery.select_batch([0.1, 0.9, 0.9, 0.2, 0.0], [0, 1, 2, 0, 0], cand, 3, rng)
    assert list(picked) == [9, 7, 11]


def test_loop_is_valid_resumable_and_deterministic():
    scn = _toy_scenario()
    full = discovery.run_loop(scn, "greedy_tabpfn", 3, discovery.fake_predict, rounds=3, batch=5)
    assert discovery.check_run(full, scn, rounds=3) == []
    assert not scn.is_target[[np.flatnonzero(scn.keys == k)[0] for k in full["initial"]]].any()
    part = discovery.run_loop(scn, "greedy_tabpfn", 3, discovery.fake_predict, rounds=2, batch=5)
    part["status"] = "running"
    resumed = discovery.run_loop(
        scn, "greedy_tabpfn", 3, discovery.fake_predict, part, rounds=3, batch=5
    )
    assert [r["selected"] for r in resumed["rounds"]] == [r["selected"] for r in full["rounds"]]


def test_check_run_catches_a_repeated_material():
    scn = _toy_scenario()
    rec = discovery.run_loop(scn, "random", 1, rounds=2, batch=5)
    rec["rounds"][1]["selected"][0] = dict(rec["rounds"][0]["selected"][0])
    assert any("already labeled" in p for p in discovery.check_run(rec, scn, rounds=2))


def test_random_simulation_matches_exact_expectation():
    scn = _toy_scenario()
    sims = discovery.simulate_random(scn, 400, rounds=4, batch=5)
    exact = discovery.random_expectation(scn, 20)
    assert abs(sims.mean() - exact["mean"]) <= 4 * exact["sd"] / np.sqrt(len(sims))


def test_pilot_decision_rule():
    win = discovery.pilot_decision({"ei": [5, 6, 7], "q90": [2, 3, 3]})
    assert win["chosen"] == ["ei"] and not win["tie"]
    tie = discovery.pilot_decision({"ei": [5, 5, 5], "q90": [5, 5, 4]})
    assert tie["chosen"] == ["ei", "q90"] and tie["tie"]


def test_gaussian_scores_match_numerical_integral():
    from scipy.stats import norm

    m, s, y_star, thr = np.array([100.0, 60.0]), np.array([10.0, 25.0]), 110.0, 119.0
    sc = discovery.gaussian_scores(m, s, y_star, thr)
    levels = (np.arange(200_000) + 0.5) / 200_000
    q = m[:, None] + s[:, None] * norm.ppf(levels)[None, :]
    ei_num = np.maximum(q - y_star, 0).mean(1)
    np.testing.assert_allclose(sc.ei, ei_num, rtol=2e-3)
    np.testing.assert_allclose(sc.q90, m + 1.2815516 * s, rtol=1e-6)
    np.testing.assert_allclose(sc.p_top1, 1 - norm.cdf((thr - m) / s), rtol=1e-9)


def test_run_loop_with_gaussian_predictor_is_valid():
    scn = _toy_scenario()

    def predict(X_train, y_train, X_pool, seed):
        out = discovery.fake_predict(X_train, y_train, X_pool, seed)
        return {"mean": out["mean"], "sd": np.full(len(out["mean"]), 15.0), "meta": {}}

    rec = discovery.run_loop(scn, "gp_ei", 0, predict, rounds=3)
    assert discovery.check_run(rec, scn, rounds=3) == []
    assert all("sd" in s and "ei" in s for r in rec["rounds"] for s in r["selected"])
    picked = [s["ei"] for s in rec["rounds"][0]["selected"]]
    assert picked == sorted(picked, reverse=True)
