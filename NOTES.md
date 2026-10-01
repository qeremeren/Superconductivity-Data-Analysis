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

- Pools (updated 2026-10-01): September's pool closed unused apart from Phase 0's 5 requests.
  October's pool (live reading in `results/api_usage.jsonl`, taken with
  `python -m src.budget --snapshot`) is 20M tokens, 0 used, and resets 2026-11-01, after the
  deadline, so it must cover Phases 2-6. The live daily limit is 15M tokens.
- Enforcement: `authorize()` refuses a run if the experiment is not in `PLAN`, if its ledger
  spend plus the run would pass its hard cap, if today's ledger spend plus the run would pass
  the daily cap, or if the run would eat into the last 1M tokens of the live monthly pool
  (read from the API; the run is refused if usage cannot be read), or past the live daily
  allowance. The returned `RunBudget`
  refuses any request beyond the number authorized. Every billed request is logged, before it is
  sent, to `results/api_ledger.jsonl` (committed). Phase 0's 5 requests are backfilled there.
- Assumptions behind the counts: one request per fit; the discovery loop costs one request per
  round per (seed, acquisition) trajectory regardless of pool size; 20 rounds. The pilot runs EI
  and q90 on both scenarios with 3 seeds; the main and hard runs use 10 seeds and 2 TabPFN
  acquisitions (the pilot's winner plus greedy mean), with caps that allow keeping both EI and
  q90 if the pilot is inconclusive. Phases 5-6 have no plan line yet; they come out of the
  headroom. Change `PLAN` and regenerate if these change.

<!-- budget-table:start -->
| Phase | Experiment | Planned requests | Hard cap | Tokens at cap | Pool | Basis |
|---:|---|---:|---:|---:|---|---|
| 0 | `p0_api_check` | 5 | 7 | 70,000 | Sep (closed) | spent: 2 recorded + 3 in crashed runs |
| 1 | `p1_data_audit` | 0 | 0 | 0 | - | no API calls |
| 2 | `p2_output_check` | 4 | 6 | 60,000 | Oct | main output with a midpoint grid; grid-size limit |
| 2 | `p2_random_protocol` | 50 | 60 | 600,000 | Oct | 25 random splits x 2 feature sets |
| 2 | `p2_grouped` | 50 | 60 | 600,000 | Oct | 25 grouped 2/3-1/3 splits x 2 feature sets |
| 2 | `p2_grouped_no_oxygen` | 50 | 60 | 600,000 | Oct | sensitivity: same, oxygen variants grouped |
| 2 | `p2_leave_family_out` | 6 | 8 | 80,000 | Oct | 3 held-out families x 2 feature sets |
| 3 | `p3_learning_curves` | 50 | 60 | 600,000 | Oct | 5 sizes (100-10k) x 5 seeds x 2 sets |
| 3 | `p3_quantile_vs_full` | 4 | 6 | 60,000 | Oct | 'full' vs quantile grid on 400 rows, 2 splits |
| 4 | `p4_pilot` | 240 | 260 | 2,600,000 | Oct | EI vs q90: 3 seeds x 2 acq. x 20 rounds x 2 scenarios |
| 4 | `p4_main` | 400 | 620 | 6,200,000 | Oct | top-1% target: 10 seeds x 2 acq. x 20 rounds (cap: 3) |
| 4 | `p4_hard` | 400 | 620 | 6,200,000 | Oct | non-cuprate pool and top-1% target, same design |

| Pool | Limit | Used at reading | Reading | Usable (minus 1M reserve) | Capped plan | Planned | Headroom at cap (requests) |
|---|---:|---:|---|---:|---:|---:|---:|
| Sep (closed) | 20,000,000 | 4,930,000 | 2026-09-28 | closed | - | - | - |
| Oct | 20,000,000 | 0 | 2026-10-01T21:25:20+00:00 | 19,000,000 | 17,600,000 | 12,540,000 | 1,400,000 (140) |
<!-- budget-table:end -->

## 2026-09-28 — Discovery loop design (decided, revised the same day)

- Pool: one row per scaled composition (element fractions summing to 1), Tc = median over its
  duplicate rows.
- Targets:
  - Main scenario: the top 1% of Tc in the full pool.
  - Hard scenario: the top 1% of Tc among non-cuprates, with a non-cuprate-only pool.
  - Why not 77 K: it is a far larger share of the pool than 1% (Phase 1 gives the exact base
    rate), so random search finds a hit within a few picks and cannot separate the methods.
- Start: a small random labeled set containing no target material.
- Main acquisitions (TabPFN), compared in the pilot, which picks the one for the main runs:
  - Expected improvement with reference y* = min(best Tc observed so far, top-1% threshold):
    EI(x) = E[max(Y - y*, 0)] ~ (1/L) sum_i max(Q(tau_i) - y*, 0) on the midpoint grid. Capping
    the reference at the threshold keeps EI rewarding new target-level materials after the first
    hit, instead of only materials that beat the record.
  - Upper-quantile score q90 = Q(0.9).
- Comparisons: greedy mean (TabPFN), greedy mean (XGBoost), random.
- Metrics: distinct target materials found vs experiments spent; rounds to the first hit;
  both against random search's expected tries-to-hit, (N + 1)/(K + 1) for K targets in a pool
  of N. P(Tc > 77 K) is reported each round (predicted hit probability vs realized hits) but not
  used as an acquisition: with an L-level quantile grid it cannot be resolved below about 1/L.
- All TabPFN scores in a round come from the same quantile-grid request.
- Superseded earlier the same day: target Tc > 77 K; EI over the current best; hard variant =
  "no cuprates in the initial set" (the pool still contained them, so random found them at once).

## 2026-09-28 — Phase 1: data audit

Source for every number here: `results/01_audit/summary.json` (key path in brackets), written
by `experiments/01_data_audit.py`; figures in `results/01_audit/figures/`; walkthrough in
`notebooks/01_eda.ipynb`. `make audit` reproduces all outputs byte for byte.

### Decisions (approved)
- Material = scaled composition (element fractions summing to 1, rounded to 1e-6):
  `src/data.py:composition_key`. Discovery pool Tc = median over a material's rows.
- `grouped` split = 25 seeded 2/3-1/3 row splits with whole materials on one side, mirroring
  `random`, so the only difference is grouping. Both saved in `data/splits/` (`src/splits.py`).

### Findings
- Alignment [alignment]: both files have 21,263 rows; `critical_temp` is identical in every
  row; `number_of_elements` matches the nonzero element count in every row.
- Materials [overview, duplicates]: 21,263 rows are 15,164 materials; 2,422 materials occur
  more than once and cover 8,521 rows. 351 materials appear under several formula spellings.
  The largest group is YBa2Cu3O7 with 110 rows. 1,001 rows report Tc below 1 K.
- Label noise [duplicates]: within-material Tc SD is 8.03 K pooled (median 1.30 K); the widest
  spread is 125 K (H2S: 60 to 185 K, pressure not recorded). Predicting each duplicated row by
  the mean of its other rows gives RMSE 9.45 K on those rows.
- RMSE floor [duplicates.rmse_floor_any_composition_model_K]: 4.30 K over all rows. The 81
  engineered features are identical within every material [features], so no model on these
  features (or on composition) can do better than predicting each material's mean Tc.
- Leakage [leakage], 25-split means:
  - random: 34.8% of test rows have an exact duplicate in train; 56.7% have a training
    composition within L1 0.01 (YBa2Cu3O7 vs YBa2Cu3O6.9 is 0.0072); 1-NN RMSE 11.20 K.
  - grouped: no exact duplicates by construction, but 40.7% of test rows are still within
    L1 0.01 of a training composition, and 21.4% have an oxygen-only variant in train (41.1% of
    cuprate test rows); 1-NN RMSE 11.80 K.
  - So grouping removes exact duplicates but leaves most near-duplicate leakage, and a
    memorizing 1-NN baseline loses only 0.6 K. Expect grouped and random model RMSEs to be
    close; the question is answered properly in Phase 2.
- Discovery pool before exclusions [discovery_pool.all_including_excluded,
  non_cuprate_including_excluded]:
  - All 15,164 materials: 2,626 above 77 K (17.3%; random search needs 5.8 tries on average).
    Top 1%: Tc >= 119 K, 154 materials (152 cuprates, 2 other), random ~97.8 tries.
  - Non-cuprates, 7,654 materials: top 1% is Tc >= 45.6 K, 77 materials (59 iron-based,
    18 other), random ~98.1 tries. Only 11 non-cuprates are above 77 K.
- Suspect entries [discovery_pool.excluded_materials], flagged by rule: 17
  materials. 12 are oxides with Ba or Sr but no Cu above 40 K whose formulas look like cuprates
  with Cu dropped (e.g. `Y1Ba23O` for YBa2Cu3O; `Bi2Sr2Ca2O` for Bi-2223). The rest are
  non-cuprates above 77 K. Unverified annotations: `H1Br3C61`/`H1Cl3C61` match C60/CHBr3 and
  C60/CHCl3 (the retracted 2001 Schoen reports of 117 K and 80 K); `H2S1` is a high-pressure
  result; `Na0.05W1O3` and `H1W1O3` are unconfirmed tungsten-bronze claims. Without the 17, the
  non-cuprate top 1% is Tc >= 44.0 K, 79 materials, 76 of them iron-based.
- Oxygen [oxygen]: 1,927 formulas give no oxygen amount (e.g. `Y1Ba2Cu3O`) and `unique_m.csv`
  encodes all of them as O = 1; 1,901 are cuprates (18% of cuprate rows). Their features are
  computed from the wrong stoichiometry, and grouping treats them as different materials from
  the fully specified formula. Their median Tc is 77.8 K vs 59.3 K for cuprates with an oxygen
  amount. (69.1% of cuprate rows have an integer O count, but that includes these 1,901.)
- Top feature [features.range_ThermalConductivity_mode_by_family]: 98.5% of cuprate rows have
  `range_ThermalConductivity` = 399.973 exactly (Cu minus O thermal conductivity), vs at most
  24.3% on any single value for iron-based and 3.5% for other. Consistent with the hypothesis
  that it acts as a cuprate detector; the formal test is Phase 5.

### Decisions on the open questions (2026-10-01), applied
1. Suspect entries: the 17 flagged materials are excluded from the discovery pools only and
   listed with reasons in `data/discovery_exclusions.csv` (generated by `python -m
   src.discovery` from `src/data.py:suspect_reasons`; a test checks the file matches the rule).
   The benchmarks keep every row. Pools are built by `src/discovery.py:build_pool`.
   - Main pool: 15,147 materials; 2,615 above 77 K (random 5.8 tries). Top 1%: Tc >= 119 K,
     152 materials, all cuprates, random ~99.0 tries [discovery_pool.main].
   - Hard pool (non-cuprates): 7,637 materials; none above 77 K. Top 1%: Tc >= 44.0 K,
     79 materials (76 iron-based, 3 other; Tc 44.0 to 55.6 K), random ~95.5 tries
     [discovery_pool.hard]. So the hard scenario is in effect "find the iron-based
     superconductors without having seen a cuprate".
2. Stricter split `grouped_no_oxygen` (sensitivity check; `src/splits.py`, saved in
   `data/splits/`): groups by composition with oxygen dropped, so all oxygen variants of a
   material, including formulas with no oxygen amount, stay on one side. Leakage
   [leakage.grouped_no_oxygen]: no oxygen-only variants in train by construction, but 37.3% of
   test rows still have a training composition within L1 0.01 (substitution neighbours such as
   Y0.9Ca0.1 vs Y1); 1-NN RMSE 12.82 K vs 11.80 K (grouped) and 11.20 K (random).
3. Missing-oxygen rows (1,927 rows, 1,901 cuprates) stay in the data as the paper had them.
   Reported as a limitation, and Phase 2 reports error on them separately
   (`src/data.py:oxygen_amount_missing`).

### Limitations recorded so far
- 1,927 formulas give no oxygen amount and are encoded as O = 1, so their composition and
  engineered features are wrong for the actual compound. Kept for comparability with the paper.
- The 17 excluded materials are judged by rule, not by checking each source; the rule could
  miss other parse errors (e.g. a cuprate that lost its Cu but has no Ba or Sr).
- No pressure information: H2S appears at 60 K and 185 K.
