"""TabPFN API request budget: the Phase 0-4 plan, a ledger of billed requests,
and a hard cap that refuses any run projected to exceed its allowance.

Every billed call goes through a RunBudget obtained from authorize(), which
refuses the run if
  (a) the experiment is not in PLAN, or its hard cap would be exceeded once the
      requests already in the ledger are counted,
  (b) the projected tokens plus today's ledger total exceed the daily cap, or
  (c) the projected tokens exceed the live monthly pool minus a reserve.
RunBudget.charge() then refuses any request beyond the number authorized, so a
runaway loop stops before it is billed. Requests are logged before they are
sent, so a failed call is over-counted rather than missed.

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

LEDGER = config.RESULTS_DIR / "api_ledger.jsonl"
NOTES = config.ROOT / "NOTES.md"
API_CHECK = config.RESULTS_DIR / "00_api_check.json"

# Every predict request at this dataset's sizes costs the server's minimum
# (results/00_api_check.json), so the budget is effectively a request count.
MIN_TOKENS_PER_REQUEST = 10_000
# Server limit test_set_max_rows_w_full_regression_output; the client splits
# larger output_type="full" calls into separately billed requests.
FULL_OUTPUT_ROWS_PER_REQUEST = 400
DAILY_CAP_TOKENS = 15_000_000  # account setting, raised by the user on 2026-09-28
POOL_RESERVE_TOKENS = 1_000_000  # never plan into the last 1M tokens of a monthly pool
MONTHLY_RESET = datetime(2026, 10, 1, tzinfo=UTC)

PRE, POST = "pre-Oct-1", "post-Oct-1"


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


PLAN = [
    Planned(0, "p0_api_check", 5, 7, PRE, "spent: 2 recorded + 3 in crashed runs"),
    Planned(1, "p1_data_audit", 0, 0, "-", "no API calls"),
    Planned(2, "p2_output_check", 4, 6, PRE, "main output with a midpoint grid; grid-size limit"),
    Planned(2, "p2_random_protocol", 50, 60, PRE, "25 random splits x 2 feature sets"),
    Planned(2, "p2_grouped", 50, 60, PRE, "5 folds x 5 repeats, grouped by composition, x 2 sets"),
    Planned(2, "p2_leave_family_out", 6, 8, PRE, "3 held-out families x 2 feature sets"),
    Planned(3, "p3_learning_curves", 50, 60, PRE, "5 sizes (100-10k) x 5 seeds x 2 sets"),
    Planned(3, "p3_quantile_vs_full", 4, 6, PRE, "'full' vs quantile grid on 400 rows, 2 splits"),
    Planned(4, "p4_pilot", 60, 70, PRE, "1 seed x 3 TabPFN acquisitions x 20 rounds"),
    Planned(4, "p4_standard", 600, 660, POST, "10 seeds x 3 TabPFN acquisitions x 20 rounds"),
    Planned(4, "p4_hard", 600, 660, POST, "same, no cuprates in the initial labeled set"),
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
    """Live monthly pool: {"current_usage", "usage_limit", "reset_time"}."""
    import tabpfn_client
    from tabpfn_client.client import ServiceClient

    config.require_tabpfn_token()
    return ServiceClient.get_api_usage(tabpfn_client.get_access_token())


@dataclass
class RunBudget:
    experiment: str
    max_requests: int
    tokens_per_request: int
    ledger: Path = LEDGER
    used: int = 0

    def charge(self, requests: int = 1, **detail) -> None:
        """Call immediately before sending `requests` billed requests."""
        if self.used + requests > self.max_requests:
            raise BudgetExceeded(
                f"{self.experiment}: run authorized for {self.max_requests} requests, "
                f"{self.used} used, {requests} more requested"
            )
        append_ledger(
            {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "experiment": self.experiment,
                "requests": requests,
                "tokens_estimated": requests * self.tokens_per_request,
                **detail,
            },
            self.ledger,
        )
        self.used += requests


def authorize(
    experiment: str,
    n_requests: int,
    tokens_per_request: int,
    *,
    usage: dict | None = None,
    ledger: Path = LEDGER,
    now: datetime | None = None,
) -> RunBudget:
    """Refuse (raise BudgetExceeded) unless the whole run fits every cap."""
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
    limit, used = int(usage["usage_limit"]), int(usage["current_usage"])
    if limit != -1 and projected > limit - used - POOL_RESERVE_TOKENS:
        raise BudgetExceeded(
            f"{experiment}: {projected:,} projected tokens exceed the monthly pool's "
            f"{limit - used:,} remaining minus the {POOL_RESERVE_TOKENS:,} reserve "
            f"(resets {usage.get('reset_time')})"
        )
    return RunBudget(experiment, n_requests, tokens_per_request, ledger)


def pool_snapshot() -> dict[str, tuple[int, int]]:
    """(limit, used) per pool. Pre-Oct-1 comes from the last recorded usage in
    results/00_api_check.json; post-Oct-1 assumes the same monthly limit, unused."""
    report = json.loads(API_CHECK.read_text())
    m = re.search(r"used (\d+) of the allowed limit of (\d+)", report["usage_after"])
    used, limit = int(m.group(1)), int(m.group(2))
    return {PRE: (limit, used), POST: (limit, 0)}


def render_table() -> str:
    rows = [
        "| Phase | Experiment | Planned requests | Hard cap | Tokens at cap | Pool | Basis |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for p in PLAN:
        tokens = p.cap * MIN_TOKENS_PER_REQUEST
        rows.append(
            f"| {p.phase} | `{p.experiment}` | {p.requests:,} | {p.cap:,} | {tokens:,} "
            f"| {p.pool} | {p.basis} |"
        )

    snapshot = pool_snapshot()
    rows += [
        "",
        "| Pool | Limit | Used at snapshot | Usable (minus 1M reserve) | Capped plan "
        "| Headroom | Headroom (requests) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pool, (limit, used) in snapshot.items():
        usable = limit - used - POOL_RESERVE_TOKENS
        capped = sum(p.cap for p in PLAN if p.pool == pool) * MIN_TOKENS_PER_REQUEST
        if pool == PRE:  # Phase 0 is already inside "used"
            capped -= PLAN_BY_NAME["p0_api_check"].cap * MIN_TOKENS_PER_REQUEST
        headroom = usable - capped
        rows.append(
            f"| {pool} | {limit:,} | {used:,} | {usable:,} | {capped:,} | {headroom:,} "
            f"| {headroom // MIN_TOKENS_PER_REQUEST:,} |"
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
    args = parser.parse_args()
    if args.update_notes:
        update_notes()
        print(f"Updated the budget table in {NOTES}")
    else:
        print(render_table())
