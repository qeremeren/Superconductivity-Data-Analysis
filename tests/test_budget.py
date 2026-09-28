from datetime import UTC, datetime

import pytest

from src import budget
from src.budget import BudgetExceeded, authorize

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)
PLENTY = {"current_usage": "0", "usage_limit": "20000000", "reset_time": "2026-10-01"}


@pytest.fixture
def ledger(tmp_path):
    return tmp_path / "ledger.jsonl"


def spend(ledger, experiment, requests, ts="2026-09-29T10:00:00+00:00", tokens=None):
    tokens = requests * budget.MIN_TOKENS_PER_REQUEST if tokens is None else tokens
    budget.append_ledger(
        {"ts": ts, "experiment": experiment, "requests": requests, "tokens_estimated": tokens},
        ledger,
    )


def test_plan_is_consistent():
    names = [p.experiment for p in budget.PLAN]
    assert len(names) == len(set(names))
    for p in budget.PLAN:
        assert p.cap >= p.requests >= 0
        assert p.pool in (budget.PRE, budget.POST, "-")


def test_full_output_is_billed_per_400_rows():
    assert budget.requests_for_predict(400, "full") == 1
    assert budget.requests_for_predict(401, "full") == 2
    assert budget.requests_for_predict(7088, "full") == 18
    assert budget.requests_for_predict(21_163, "quantiles") == 1
    assert budget.requests_for_predict(21_163, "main") == 1


def test_unknown_experiment_is_refused(ledger):
    with pytest.raises(BudgetExceeded, match="not in budget.PLAN"):
        authorize("p9_surprise", 1, 10_000, usage=PLENTY, ledger=ledger, now=NOW)


def test_experiment_cap_counts_earlier_spend(ledger):
    cap = budget.PLAN_BY_NAME["p2_output_check"].cap
    spend(ledger, "p2_output_check", cap - 1)
    authorize("p2_output_check", 1, 10_000, usage=PLENTY, ledger=ledger, now=NOW)
    with pytest.raises(BudgetExceeded, match="hard cap"):
        authorize("p2_output_check", 2, 10_000, usage=PLENTY, ledger=ledger, now=NOW)


def test_daily_cap_counts_todays_spend_only(ledger):
    near_cap = budget.DAILY_CAP_TOKENS - 50_000
    spend(ledger, "p4_pilot", 1, ts="2026-09-28T23:00:00+00:00", tokens=near_cap)
    authorize("p4_standard", 10, 10_000, usage=PLENTY, ledger=ledger, now=NOW)
    spend(ledger, "p4_pilot", 1, ts="2026-09-29T01:00:00+00:00", tokens=near_cap)
    with pytest.raises(BudgetExceeded, match="daily cap"):
        authorize("p4_standard", 10, 10_000, usage=PLENTY, ledger=ledger, now=NOW)


def test_monthly_pool_keeps_a_reserve(ledger):
    limit = 20_000_000
    enough = {"current_usage": limit - budget.POOL_RESERVE_TOKENS - 100_000, "usage_limit": limit}
    authorize("p4_pilot", 10, 10_000, usage=enough, ledger=ledger, now=NOW)
    short = {"current_usage": limit - budget.POOL_RESERVE_TOKENS - 50_000, "usage_limit": limit}
    with pytest.raises(BudgetExceeded, match="monthly pool"):
        authorize("p4_pilot", 10, 10_000, usage=short, ledger=ledger, now=NOW)


def test_unlimited_pool_passes(ledger):
    unlimited = {"current_usage": 10**12, "usage_limit": -1}
    authorize("p4_pilot", 10, 10_000, usage=unlimited, ledger=ledger, now=NOW)


def test_unreadable_usage_fails_closed(ledger, monkeypatch):
    def broken():
        raise ConnectionError("offline")

    monkeypatch.setattr(budget, "fetch_usage", broken)
    with pytest.raises(BudgetExceeded, match="could not read live API usage"):
        authorize("p4_pilot", 1, 10_000, ledger=ledger, now=NOW)


def test_projection_never_below_minimum_charge(ledger):
    run = authorize("p4_pilot", 1, 5, usage=PLENTY, ledger=ledger, now=NOW)
    assert run.tokens_per_request == budget.MIN_TOKENS_PER_REQUEST


def test_run_refuses_requests_beyond_authorization_and_logs_first(ledger):
    run = authorize("p2_output_check", 2, 10_000, usage=PLENTY, ledger=ledger, now=NOW)
    run.charge(output_type="main")
    run.charge(output_type="main")
    with pytest.raises(BudgetExceeded, match="authorized for 2"):
        run.charge()
    entries = budget.read_ledger(ledger)
    assert [e["requests"] for e in entries] == [1, 1]
    assert all(e["experiment"] == "p2_output_check" for e in entries)
    assert entries[0]["output_type"] == "main"


def test_committed_ledger_is_within_plan():
    spent = {}
    for e in budget.read_ledger():
        spent[e["experiment"]] = spent.get(e["experiment"], 0) + e["requests"]
    for experiment, requests in spent.items():
        assert experiment in budget.PLAN_BY_NAME
        assert requests <= budget.PLAN_BY_NAME[experiment].cap


def test_notes_table_matches_plan():
    notes = budget.NOTES.read_text()
    section = notes.split(budget.START)[1].split(budget.END)[0].strip()
    assert section == budget.render_table(), "run `python -m src.budget --update-notes`"
