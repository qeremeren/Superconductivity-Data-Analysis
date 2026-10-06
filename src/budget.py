"""TabPFN API request budget: the Phase 0-4 plan, a ledger of billed requests,
and a hard cap that refuses any run projected to exceed its allowance.

Every billed call goes through a RunBudget obtained from authorize(), which
refuses the run if
  (a) the experiment is not in PLAN, or its hard cap would be exceeded once the
      requests already in this clone's ledger are counted,
  (b) the projected tokens plus today's ledger total exceed the daily cap, or the
      live daily allowance of the caller's account, or
  (c) the projected tokens exceed the caller's live monthly pool minus a reserve.
RunBudget.charge() then refuses any request beyond the number authorized, so a
runaway loop stops before it is billed. Requests are logged before they are
sent, so a failed call is over-counted rather than missed.

Two files, so nobody's spend counts against anyone else's re-run:
  LEDGER        .budget/ledger.jsonl, per clone and gitignored. The only file the
                guard reads.
  SPEND_RECORD  results/api_ledger.jsonl, committed: the project author's record of
                billed requests, for provenance. Appended to only when
                TABPFN_RECORD_SPEND=1; never read by the guard.

`python -m src.budget` prints the plan; `--update-notes` rewrites it in NOTES.md.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src import config

LEDGER = config.ROOT / ".budget" / "ledger.jsonl"
SPEND_RECORD = config.RESULTS_DIR / "api_ledger.jsonl"
USAGE_LOG = config.RESULTS_DIR / "api_usage.jsonl"  # live usage readings, via --snapshot
NOTES = config.ROOT / "NOTES.md"
API_CHECK = config.RESULTS_DIR / "00_api_check.json"

# Every predict request at this dataset's sizes costs the server's minimum
# (results/00_api_check.json), so the budget is effectively a request count.
MIN_TOKENS_PER_REQUEST = 10_000
# Server limit test_set_max_rows_w_full_regression_output; the client splits
# larger output_type="full" calls into separately billed requests.
FULL_OUTPUT_ROWS_PER_REQUEST = 400
DAILY_CAP_TOKENS = 15_000_000  # account setting; the live daily limit is checked too
POOL_RESERVE_TOKENS = 1_000_000  # never plan into the last 1M tokens of a monthly pool

# Monthly pools. September's closed on 2026-10-01; October's resets 2026-11-01, after
# the 2026-10-06 deadline, so it is the only pool left for Phases 2-6.
SEP, OCT = "Sep (closed)", "Oct"


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Planned:
    phase: int
    experiment: str
    requests: int
    cap: int  # hard cap, with headroom for reruns
    pool: str
    basis: str
    tokens_per_request: int = MIN_TOKENS_PER_REQUEST  # planning figure for the table


PLAN = [
    Planned(0, "p0_api_check", 5, 7, SEP, "spent: 2 recorded + 3 in crashed runs"),
    Planned(1, "p1_data_audit", 0, 0, "-", "no API calls"),
    Planned(2, "p2_output_check", 4, 6, OCT, "main output with a midpoint grid; grid-size limit"),
    Planned(2, "p2_random_protocol", 50, 60, OCT, "25 random splits x 2 feature sets"),
    Planned(2, "p2_grouped", 50, 60, OCT, "25 grouped 2/3-1/3 splits x 2 feature sets"),
    Planned(2, "p2_grouped_no_oxygen", 50, 60, OCT, "sensitivity: same, oxygen variants grouped"),
    Planned(2, "p2_leave_family_out", 6, 8, OCT, "3 held-out families x 2 feature sets"),
    # Thinking: a fit (~14.6k tokens, medium effort) plus a predict (~59.7k) per split, so
    # two requests averaging ~37.5k tokens (estimate_cost quotes, 2026-10-02).
    Planned(2, "p2_thinking_probe", 2, 3, OCT, "Thinking + group_col, grouped split 0", 37_500),
    Planned(
        2, "p2_thinking", 20, 22, OCT, "Thinking add-on: grouped splits 0-9, composition", 37_500
    ),
    Planned(3, "p3_learning_curves", 50, 60, OCT, "5 sizes (100-10k) x 5 seeds x 2 sets"),
    Planned(3, "p3_quantile_vs_full", 4, 6, OCT, "'full' vs quantile grid on 400 rows, 2 splits"),
    Planned(4, "p4_grid_check", 2, 3, OCT, "tail-dense vs 999-level grid, one pool"),
    Planned(4, "p4_pilot", 240, 260, OCT, "EI vs q90: 3 seeds x 2 acq. x 20 rounds x 2 scenarios"),
    Planned(4, "p4_main", 400, 620, OCT, "top-1% target: 10 seeds x 2 acq. x 20 rounds (cap: 3)"),
    Planned(4, "p4_hard", 400, 620, OCT, "non-cuprate pool and top-1% target, same design"),
    Planned(6, "p6_demo", 20, 40, OCT, "demo: 20 showcase formulas, one request each"),
]
PLAN_BY_NAME = {p.experiment: p for p in PLAN}


def requests_for_predict(n_test_rows: int, output_type: str) -> int:
    if output_type == "full":
        return math.ceil(n_test_rows / FULL_OUTPUT_ROWS_PER_REQUEST)
    return 1


def read_ledger(ledger: Path = LEDGER) -> list[dict]:
    if not ledger.is_file():
        return []
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]


def append_ledger(entry: dict, ledger: Path = LEDGER) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def estimate_tokens(X_train, X_test) -> int:
    """Projected tokens for one predict request (free: sends only the shapes)."""
    from tabpfn_client import estimate_cost

    quote = estimate_cost(X_train, X_test, model_version=config.TABPFN_MODEL_VERSION)
    return max(MIN_TOKENS_PER_REQUEST, int(quote.estimated_cost))


def fetch_usage() -> dict:
    """Live usage: monthly_* and daily_* token limits and use, reset times (free call)."""
    import tabpfn_client
    from tabpfn_client.client import ServiceClient

    config.require_tabpfn_token()
    return ServiceClient.get_api_usage(tabpfn_client.get_access_token())


def record_usage(usage: dict | None = None, log: Path = USAGE_LOG) -> dict:
    usage = usage if usage is not None else fetch_usage()
    entry = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **usage}
    append_ledger(entry, log)
    return entry


def _monthly(usage: dict) -> tuple[int, int]:
    limit = usage.get("monthly_token_limit", usage.get("usage_limit"))
    used = usage.get("monthly_tokens_used", usage.get("current_usage"))
    return int(limit), int(used)


@dataclass
class RunBudget:
    experiment: str
    max_requests: int
    tokens_per_request: int
    ledger: Path = LEDGER
    record: Path | None = None
    used: int = 0

    def charge(self, requests: int = 1, **detail) -> None:
        """Call immediately before sending `requests` billed requests."""
        if self.used + requests > self.max_requests:
            raise BudgetExceeded(
                f"{self.experiment}: run authorized for {self.max_requests} requests, "
                f"{self.used} used, {requests} more requested"
            )
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "experiment": self.experiment,
            "requests": requests,
            "tokens_estimated": requests * self.tokens_per_request,
            **detail,
        }
        append_ledger(entry, self.ledger)
        if self.record is not None:
            append_ledger(entry, self.record)
        self.used += requests


def authorize(
    experiment: str,
    n_requests: int,
    tokens_per_request: int,
    *,
    usage: dict | None = None,
    ledger: Path = LEDGER,
    record: Path | None = None,
    now: datetime | None = None,
) -> RunBudget:
    """Refuse (raise BudgetExceeded) unless the whole run fits every cap.

    With the default ledger and TABPFN_RECORD_SPEND=1, charges are also appended to
    the committed SPEND_RECORD. A ledger passed explicitly (tests) never records.
    """
    if record is None and ledger == LEDGER and config.record_spend():
        record = SPEND_RECORD
    now = now or datetime.now(UTC)
    plan = PLAN_BY_NAME.get(experiment)
    if plan is None:
        raise BudgetExceeded(f"{experiment!r} is not in budget.PLAN; add it there first")

    entries = read_ledger(ledger)
    spent = sum(e["requests"] for e in entries if e["experiment"] == experiment)
    if spent + n_requests > plan.cap:
        raise BudgetExceeded(
            f"{experiment}: {spent} requests already spent + {n_requests} planned "
            f"exceeds the hard cap of {plan.cap}"
        )

    tokens_per_request = max(tokens_per_request, MIN_TOKENS_PER_REQUEST)
    projected = n_requests * tokens_per_request
    today = sum(e["tokens_estimated"] for e in entries if e["ts"][:10] == now.date().isoformat())
    if today + projected > DAILY_CAP_TOKENS:
        raise BudgetExceeded(
            f"{experiment}: {projected:,} projected + {today:,} already today exceeds "
            f"the daily cap of {DAILY_CAP_TOKENS:,} tokens"
        )

    if usage is None:
        try:
            usage = fetch_usage()
        except Exception as exc:
            raise BudgetExceeded(f"could not read live API usage, refusing: {exc}") from exc
    if "daily_token_limit" in usage and int(usage["daily_token_limit"]) != -1:
        left_today = int(usage["daily_token_limit"]) - int(usage["daily_tokens_used"])
        if projected > left_today:
            raise BudgetExceeded(
                f"{experiment}: {projected:,} projected tokens exceed the live daily "
                f"allowance of {left_today:,} left today"
            )
    limit, used = _monthly(usage)
    if limit != -1 and projected > limit - used - POOL_RESERVE_TOKENS:
        raise BudgetExceeded(
            f"{experiment}: {projected:,} projected tokens exceed the monthly pool's "
            f"{limit - used:,} remaining minus the {POOL_RESERVE_TOKENS:,} reserve "
            f"(resets {usage.get('reset_time')})"
        )
    return RunBudget(experiment, n_requests, tokens_per_request, ledger, record)


def pool_snapshot() -> dict[str, tuple[int, int, str]]:
    """(limit, used, as-of) per pool. September: the last reading in
    results/00_api_check.json. October: the latest reading in results/api_usage.jsonl
    taken after the reset (`python -m src.budget --snapshot`)."""
    report = json.loads(API_CHECK.read_text())
    m = re.search(r"used (\d+) of the allowed limit of (\d+)", report["usage_after"])
    sep = (int(m.group(2)), int(m.group(1)), report["created_at"][:10])
    october = [e for e in read_ledger(USAGE_LOG) if e["ts"] >= "2026-10-01"]
    if not october:
        raise RuntimeError("no October usage reading; run `python -m src.budget --snapshot`")
    latest = october[-1]
    return {SEP: sep, OCT: (*_monthly(latest), latest["ts"])}


def spent_by_experiment(record: Path = SPEND_RECORD) -> dict[str, int]:
    """Requests per experiment in the committed spend record. The NOTES table uses this,
    not the per-clone ledger, so it renders the same in every clone."""
    spent: dict[str, int] = {}
    for e in read_ledger(record):
        spent[e["experiment"]] = spent.get(e["experiment"], 0) + e["requests"]
    return spent


def render_table() -> str:
    spent = spent_by_experiment()
    rows = [
        "| Phase | Experiment | Planned requests | Hard cap | Spent | Tokens at cap | Pool "
        "| Basis |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for p in PLAN:
        tokens = p.cap * p.tokens_per_request
        rows.append(
            f"| {p.phase} | `{p.experiment}` | {p.requests:,} | {p.cap:,} "
            f"| {spent.get(p.experiment, 0):,} | {tokens:,} | {p.pool} | {p.basis} |"
        )

    snapshot = pool_snapshot()
    rows += [
        "",
        'Spend already made is inside "used"; the plan columns count only what is left of each',
        "experiment's cap and plan.",
        "",
        "| Pool | Limit | Used at reading | Reading | Usable (minus 1M reserve) "
        "| Remaining caps | Remaining plan | Headroom at cap (requests) |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for pool, (limit, used, as_of) in snapshot.items():
        if pool == SEP:
            rows.append(f"| {pool} | {limit:,} | {used:,} | {as_of} | closed | - | - | - |")
            continue
        in_pool = [p for p in PLAN if p.pool == pool]
        left_cap = sum(
            max(p.cap - spent.get(p.experiment, 0), 0) * p.tokens_per_request for p in in_pool
        )
        left_plan = sum(
            max(p.requests - spent.get(p.experiment, 0), 0) * p.tokens_per_request for p in in_pool
        )
        usable = limit - used - POOL_RESERVE_TOKENS
        headroom = usable - left_cap
        rows.append(
            f"| {pool} | {limit:,} | {used:,} | {as_of} | {usable:,} | {left_cap:,} "
            f"| {left_plan:,} | {headroom:,} ({headroom // MIN_TOKENS_PER_REQUEST:,}) |"
        )
    return "\n".join(rows)


START, END = "<!-- budget-table:start -->", "<!-- budget-table:end -->"


def update_notes(notes: Path = NOTES) -> None:
    text = notes.read_text()
    before, rest = text.split(START)
    _, after = rest.split(END)
    notes.write_text(f"{before}{START}\n{render_table()}\n{END}{after}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the API request budget plan.")
    parser.add_argument("--update-notes", action="store_true", help="rewrite it in NOTES.md")
    parser.add_argument(
        "--snapshot", action="store_true", help="append a live usage reading (free call)"
    )
    args = parser.parse_args()
    if args.snapshot:
        print(record_usage())
    if args.update_notes:
        update_notes()
        print(f"Updated the budget table in {NOTES}")
    else:
        print(render_table())
