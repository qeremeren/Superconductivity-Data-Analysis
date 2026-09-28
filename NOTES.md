# NOTES

Running log of decisions and findings. The README is written from this file at the end.

## 2026-09-28 — Phase 0: setup

### Environment
- Python 3.12 via uv (`.python-version`, `uv.lock`); `requirements.txt` is exported from the lock.
  System Python on the dev machine is 3.9, which tabpfn-client does not support (needs >=3.10).
- tabpfn-client pinned to 0.6.1. It caps pandas at <=2.3.3.
- macOS: the xgboost wheel needs the OpenMP runtime (`brew install libomp`).
- Scripts run as modules from the repo root (`python -m src.data`, `python -m experiments.00_check_api`)
  so `src` is importable without installing the project.

### TabPFN API (from docs.priorlabs.ai and the tabpfn-client README/source, read 2026-09-28)
- We pin `model_path="v3.5_default"` instead of the default `"auto"`, so a server-side model change
  before the deadline cannot silently change results. Through the API this is **TabPFN-3.5-Plus**
  (base TabPFN-3.5 plus proprietary text processing). All our features are numeric, so the text
  processing should not matter, but results must be labelled TabPFN-3.5-Plus (API).
- Each estimator exposes `last_meta` with the server-reported model version; record it with every
  cached result.
- `random_state` defaults to 0 in the client; we pass it explicitly.
- Regression `predict(..., output_type="full")` returns `logits` (n, n_buckets) and bucket `borders`
  as NumPy arrays. That is enough to compute any quantile, P(Tc > 77 K) and CRPS client-side,
  so no local `tabpfn` package, torch or model weights are needed. "full" has a stricter cap on
  test rows per call than plain predictions; chunk large test sets.
- Metering: predictions cost tokens; uploads and standard fits are free; minimum 10k tokens per
  billable request. Default budget 5M tokens/day and 20M/month (monthly resets on the 1st, UTC).
  The account's daily cap was raised to 15M on 2026-09-28. `estimate_cost()` is free (sends only
  dimensions). Rate limits: 60 predict and 60 fit requests per minute.
- tabpfn-client (via tabpfn-common-utils) sends PostHog usage telemetry by default outside CI;
  opt out with `TABPFN_DISABLE_TELEMETRY=1` (see `.env.example`).
- The token is read from `TABPFN_TOKEN` at call time; we load a git-ignored `.env` with
  python-dotenv. Live calls are opt-in (`TABPFN_LIVE=1` or `--live`).

### Data
- UCI dataset id 464, "Superconductivty Data", license CC BY 4.0. Downloaded as a zip; the two
  CSVs are verified against SHA-256 hashes pinned in `src/data.py`.
- `train.csv`: 21,263 x 82 (81 engineered features + `critical_temp`).
  `unique_m.csv`: 21,263 x 88 (86 element columns H..Rn + `critical_temp` + `material`).
  No missing values in either file.
- Row alignment verified: `critical_temp` is identical row by row, and `number_of_elements` in
  `train.csv` equals the number of nonzero element columns in `unique_m.csv` for all 21,263 rows.
  (Covered by `tests/test_data.py`.)
- Citation (checked on Crossref): K. Hamidieh, Computational Materials Science 154 (2018) 346–354,
  doi:10.1016/j.commatsci.2018.07.052.

### Live API check (`results/00_api_check.json`, raw arrays in `results/00_api_check_outputs.npz`)
- Server metadata for `model_path="v3.5_default"`: checkpoint `tabpfn-v3.5-20260909.safetensors`,
  tabpfn package 9.0.0, `n_estimators=8`, `TRANSFORM_TEXT=False`. That is the same checkpoint as the
  open-weights base TabPFN-3.5; with numeric-only data the Plus text processing is switched off.
  Report the model as "TabPFN-3.5 (checkpoint 20260909, via the Prior Labs API)".
- The docs FAQ says `model.last_meta`, but tabpfn-client 0.6.1 only has the private `_last_meta`.
- Cost: `estimate_cost()` gives the 10k-token minimum for every size we will use (full random
  split 14,175 x 81 -> 7,088; discovery round 100 -> 21,163). Actual charge for the two live
  predictions was exactly 20,000 tokens, matching the estimate. **Budget is a request count, not
  a data size**: ~10k tokens per predict request.
- `get_api_usage()` reports the monthly pool: 20M, resets 2026-10-01 00:00 UTC; 4.93M used after
  this check. (About 30k of that went to two runs of this script that crashed after the API
  calls; the script now saves raw outputs before analysing and supports `--from-saved`.)
- Server limit `test_set_max_rows_w_full_regression_output = 400`. The client splits larger
  `output_type="full"` calls into several requests, each billed separately: a 7,088-row test set
  would cost 18 requests. Other output types allow 1M test rows per request.
- `output_type="full"` returns `logits` (n, 5000) and `borders` (5001,), with borders spanning
  about -4611 to 4679 K (the fixed z-space grid rescaled to the target), plus mean/median/mode
  and 9 decile quantiles.
- Our client-side reconstruction from logits and borders (softmax, then invert the
  piecewise-uniform CDF) matches the server: mean within 0.01 K, all 99 quantile levels within
  4e-4 K. A single `output_type="quantiles"` request with 99 levels (0.01..0.99) is accepted and
  equals the full output's deciles exactly, so repeated requests with `random_state=0` are
  deterministic.

### Decisions for later phases
- Benchmark and discovery predictions use one request per fit with a dense quantile grid, not
  `"full"` (400-row cap). Prefer `output_type="main"` if it accepts a custom grid, so the
  server's mean arrives in the same request; `p2_output_check` verifies this.
- Use a midpoint grid tau_i = (i - 0.5)/L with L = 100 (0.005 ... 0.995), so expectations over the
  predictive distribution are a midpoint rule: E[g(Y)] ~ (1/L) sum_i g(Q(tau_i)). Try a larger L
  in `p2_output_check` if the server accepts it.
- CRPS from the quantile-score integral over the grid. P(Tc > 77 K) from the grid is only
  resolvable to about 1/L; Phase 3 measures the approximation error against `"full"`.
- Cache per-row quantile grids (small); full logits only for a handful of demo rows
  (100 rows of float64 logits compress to ~1.5 MB).

## 2026-09-28 — API request budget (Phases 0-4)

Generated by `python -m src.budget --update-notes` from `src/budget.py:PLAN`, the same plan the
hard cap enforces (a test fails if this table and the code drift apart). Tokens assume the
10k-token minimum per predict request, which is what every size we use costs.

- Pools: "pre-Oct-1" is the current monthly pool, which resets 2026-10-01 00:00 UTC
  (2026-09-30 17:00 PDT); whatever is unused then is lost. "post-Oct-1" assumes the monthly
  limit stays 20M; confirm on Oct 1. The account's daily cap is 15M tokens.
- Enforcement: `authorize()` refuses a run if the experiment is not in `PLAN`, if its ledger
  spend plus the run would pass its hard cap, if today's ledger spend plus the run would pass
  the daily cap, or if the run would eat into the last 1M tokens of the live monthly pool
  (read from the API; the run is refused if usage cannot be read). The returned `RunBudget`
  refuses any request beyond the number authorized. Every billed request is logged, before it is
  sent, to `results/api_ledger.jsonl` (committed). Phase 0's 5 requests are backfilled there.
- Assumptions behind the counts: one request per fit; the discovery loop costs one request per
  round per (seed, acquisition) trajectory regardless of pool size; 20 rounds, 10 seeds, and 3
  TabPFN acquisitions (EI, q90, greedy mean). Change `PLAN` and regenerate if these change.

<!-- budget-table:start -->
| Phase | Experiment | Planned requests | Hard cap | Tokens at cap | Pool | Basis |
|---:|---|---:|---:|---:|---|---|
| 0 | `p0_api_check` | 5 | 7 | 70,000 | pre-Oct-1 | spent: 2 recorded + 3 in crashed runs |
| 1 | `p1_data_audit` | 0 | 0 | 0 | - | no API calls |
| 2 | `p2_output_check` | 4 | 6 | 60,000 | pre-Oct-1 | main output with a midpoint grid; grid-size limit |
| 2 | `p2_random_protocol` | 50 | 60 | 600,000 | pre-Oct-1 | 25 random splits x 2 feature sets |
| 2 | `p2_grouped` | 50 | 60 | 600,000 | pre-Oct-1 | 5 folds x 5 repeats, grouped by composition, x 2 sets |
| 2 | `p2_leave_family_out` | 6 | 8 | 80,000 | pre-Oct-1 | 3 held-out families x 2 feature sets |
| 3 | `p3_learning_curves` | 50 | 60 | 600,000 | pre-Oct-1 | 5 sizes (100-10k) x 5 seeds x 2 sets |
| 3 | `p3_quantile_vs_full` | 4 | 6 | 60,000 | pre-Oct-1 | 'full' vs quantile grid on 400 rows, 2 splits |
| 4 | `p4_pilot` | 60 | 70 | 700,000 | pre-Oct-1 | 1 seed x 3 TabPFN acquisitions x 20 rounds |
| 4 | `p4_standard` | 600 | 660 | 6,600,000 | post-Oct-1 | 10 seeds x 3 TabPFN acquisitions x 20 rounds |
| 4 | `p4_hard` | 600 | 660 | 6,600,000 | post-Oct-1 | same, no cuprates in the initial labeled set |

| Pool | Limit | Used at snapshot | Usable (minus 1M reserve) | Capped plan | Headroom | Headroom (requests) |
|---|---:|---:|---:|---:|---:|---:|
| pre-Oct-1 | 20,000,000 | 4,930,000 | 14,070,000 | 2,700,000 | 11,370,000 | 1,137 |
| post-Oct-1 | 20,000,000 | 0 | 19,000,000 | 13,200,000 | 5,800,000 | 580 |
<!-- budget-table:end -->

## 2026-09-28 — Discovery loop design (decided)

- Main acquisitions (TabPFN):
  - Expected improvement over the best Tc observed so far in the labeled set:
    EI(x) = E[max(Y - y*, 0)] ~ (1/L) sum_i max(Q(tau_i) - y*, 0) on the midpoint grid.
  - Upper-quantile score q90 = Q(0.9).
- Comparisons: greedy mean (TabPFN), greedy mean (XGBoost), random.
- P(Tc > 77 K) is reported each round (predicted hit probability vs realized hits) but not used
  as an acquisition: with an L-level quantile grid it cannot be resolved below about 1/L (1% at
  99 levels), so most of the pool would tie.
- All TabPFN scores in a round come from the same quantile-grid request.
- Open, to decide in Phase 4 with Phase 1's numbers:
  - Incumbent after the first >77 K hit. EI over the current best then rewards beating the
    record, while the headline metric counts distinct >77 K materials. Option: use
    y* = min(current best, 77 K).
  - Difficulty. If >77 K materials are common in the deduplicated pool, random search finds one
    within a few picks and "rounds to first hit" cannot separate the methods. Phase 1 measures
    the base rates.
