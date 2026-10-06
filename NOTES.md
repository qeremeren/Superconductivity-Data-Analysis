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
  refuses any request beyond the number authorized.
- Ledgers (changed 2026-10-02 for Phase 6 reproducibility): the guard reads only the per-clone,
  gitignored `.budget/ledger.jsonl`, so the author's spend never counts against someone else's
  re-run; their own account limits come from the live API. Every billed request is logged
  there before it is sent. The committed `results/api_ledger.jsonl` is the author's provenance
  record, appended only when `TABPFN_RECORD_SPEND=1` (set in the author's `.env`) and never
  read by the guard. Phase 0's 5 requests are in both.
- Assumptions behind the counts: one request per fit; the discovery loop costs one request per
  round per (seed, acquisition) trajectory regardless of pool size; 20 rounds. The pilot runs EI
  and q90 on both scenarios with 3 seeds; the main and hard runs use 10 seeds and 2 TabPFN
  acquisitions (the pilot's winner plus greedy mean), with caps that allow keeping both EI and
  q90 if the pilot is inconclusive. Phases 5-6 have no plan line yet; they come out of the
  headroom. Change `PLAN` and regenerate if these change.

<!-- budget-table:start -->
| Phase | Experiment | Planned requests | Hard cap | Spent | Tokens at cap | Pool | Basis |
|---:|---|---:|---:|---:|---:|---|---|
| 0 | `p0_api_check` | 5 | 7 | 5 | 70,000 | Sep (closed) | spent: 2 recorded + 3 in crashed runs |
| 1 | `p1_data_audit` | 0 | 0 | 0 | 0 | - | no API calls |
| 2 | `p2_output_check` | 4 | 6 | 4 | 60,000 | Oct | main output with a midpoint grid; grid-size limit |
| 2 | `p2_random_protocol` | 50 | 60 | 50 | 600,000 | Oct | 25 random splits x 2 feature sets |
| 2 | `p2_grouped` | 50 | 60 | 50 | 600,000 | Oct | 25 grouped 2/3-1/3 splits x 2 feature sets |
| 2 | `p2_grouped_no_oxygen` | 50 | 60 | 50 | 600,000 | Oct | sensitivity: same, oxygen variants grouped |
| 2 | `p2_leave_family_out` | 6 | 8 | 6 | 80,000 | Oct | 3 held-out families x 2 feature sets |
| 2 | `p2_thinking_probe` | 2 | 3 | 2 | 112,500 | Oct | Thinking + group_col, grouped split 0 |
| 2 | `p2_thinking` | 20 | 22 | 20 | 825,000 | Oct | Thinking add-on: grouped splits 0-9, composition |
| 3 | `p3_learning_curves` | 50 | 60 | 50 | 600,000 | Oct | 5 sizes (100-10k) x 5 seeds x 2 sets |
| 3 | `p3_quantile_vs_full` | 4 | 6 | 2 | 60,000 | Oct | 'full' vs quantile grid on 400 rows, 2 splits |
| 4 | `p4_grid_check` | 2 | 3 | 2 | 30,000 | Oct | tail-dense vs 999-level grid, one pool |
| 4 | `p4_pilot` | 240 | 260 | 240 | 2,600,000 | Oct | EI vs q90: 3 seeds x 2 acq. x 20 rounds x 2 scenarios |
| 4 | `p4_main` | 400 | 620 | 400 | 6,200,000 | Oct | top-1% target: 10 seeds x 2 acq. x 20 rounds (cap: 3) |
| 4 | `p4_hard` | 400 | 620 | 404 | 6,200,000 | Oct | non-cuprate pool and top-1% target, same design |
| 6 | `p6_demo` | 20 | 40 | 20 | 400,000 | Oct | demo: 20 showcase formulas, one request each |

Spend already made is inside "used"; the plan columns count only what is left of each
experiment's cap and plan.

| Pool | Limit | Used at reading | Reading | Usable (minus 1M reserve) | Remaining caps | Remaining plan | Headroom at cap (requests) |
|---|---:|---:|---|---:|---:|---:|---:|
| Sep (closed) | 20,000,000 | 4,930,000 | 2026-09-28 | closed | - | - | - |
| Oct | 60,000,000 | 13,804,949 | 2026-10-06T18:35:44+00:00 | 45,195,051 | 5,362,500 | 20,000 | 39,832,551 (3,983) |
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

## 2026-10-02 — Phase 2, steps 1-2: local baselines and replication

Source: `results/02_benchmark/` (`replication.json`, `metrics.csv`, `per_split_metrics.parquet`),
written by `experiments/02_local_models.py` and `experiments/02_metrics.py`. No API calls.

### Decisions (approved)
- Composition features = 86 element fractions; engineered = the paper's 81 features.
- Published-settings XGBoost uses `tree_method="exact"` (closest to the paper's R xgboost);
  tuned XGBoost uses `hist`.
- Nested XGBoost tuning: per split, random search scored on an inner holdout that mirrors the
  outer split type, early stopping, refit on all training rows. Timed on random split 0
  (`experiments/02_xgb_tuning.py --time-first`): 2.51 s per trial, 2.58 s per refit; 30 trials
  on all 156 split/feature jobs projects to 3.38 h, over the ~3 h limit, so the runs use the best
  of the first 26 trials (trials are sampled in a fixed order and are independent, so this
  equals a 26-trial search; split 0's record keeps all 30).

### Replication of Hamidieh (2018) on the random split (25 splits, RMSE = sqrt(mean MSE))
- Published-settings XGBoost, engineered features: RMSE 9.42 K, R² 0.924 (paper: 9.5 K, 0.92).
- Linear regression, engineered features: RMSE 17.62 K, R² 0.735 (paper: 17.6 K, 0.74).
- Both within 1% of the published numbers, so the pipeline reproduces the paper's protocol.

### Early observations (local models only; TabPFN and tuned XGBoost not run yet)
- Grouping changes published-settings XGBoost little: 9.42 K (random) -> 9.76 K (grouped),
  engineered features. Grouping oxygen variants too costs more: 10.85 K.
- The 1-NN composition lookup scores 11.20 K on the random split, within 1.8 K of XGBoost.
- Leave-family-out errors are large for every model (e.g. XGBoost on held-out cuprates: RMSE
  46-54 K, R² below 0): composition models do not extrapolate to an unseen family.
- Check (2026-10-02): published-settings XGBoost scores 10.85 K on `grouped_no_oxygen` with both
  feature sets by coincidence of rounding: 10.8452 K (composition) vs 10.8545 K (engineered).
  The prediction files differ (no identical predictions; mean |difference| 3.7 K, correlation
  0.983) and per-split RMSE differs on all 25 splits (split-to-split SD 0.42 vs 0.77 K).

### Requirement carried into Phase 3
- Leave-family-out results must include TabPFN's interval coverage and width on the held-out
  family (50/80/90/95%), next to in-distribution coverage: "does the model know when it is
  extrapolating?" is a key question for the write-up. The committed TabPFN summaries for the
  leave-family-out folds hold the quantiles needed (`q0.025` ... `q0.975`).

## 2026-10-02 — Phase 2, step 3: API output check (4 requests, 40,000 tokens)

Source: `results/02_benchmark/output_check.json` (`experiments/02_output_check.py`); random
split 0, engineered features, 14,175 training and 7,088 test rows.
- `output_type="main"` with a custom quantile grid returns mean, median, mode and the requested
  quantiles in one request; all rows are monotone. Its mean equals a plain `output_type="mean"`
  request exactly (max |difference| 0.0 K), so one request per fit gives both.
- 107-, 199- and 999-level grids are all accepted at the same charge (10,000 tokens each).
  Wall times 10-23 s per request (63 s for 999 levels, mostly transfer).
- Consequence for Phase 4: a 999-level grid would resolve tail probabilities such as
  P(Tc > 77 K) or the top-1% exceedance to about 0.1% instead of 1%. The benchmark keeps the
  planned 107 levels.

## 2026-10-02 — Phase 2, step 4: TabPFN-3.5 on all splits (156 requests, 1.56M tokens)

Source: `results/02_benchmark/tabpfn/` (per-row summaries + request records),
`metrics.csv`, `paired.json`; `experiments/02_tabpfn.py --live --workers 4`, ~22 min.
- All 156 requests succeeded. Live usage after the run: 1,600,000 tokens this month (= 160
  ledger requests x 10,000, including the output check). `make benchmark` (no API) finds all
  156 cached with matching fingerprints; every summary's rows and true Tc match its split.
- Committed summaries total 67 MB (largest file under 1 MB), above the ~40 MB estimate.

### Interim results before tuned XGBoost finished (superseded by the step 4 summary below)
| RMSE (K) | random | grouped | grouped_no_oxygen |
|---|---:|---:|---:|
| TabPFN, composition | 8.70 | 8.97 | 9.95 |
| TabPFN, engineered | 8.72 | 9.01 | 10.54 |
| XGBoost published, engineered | 9.42 | 9.76 | 10.85 |
| XGBoost published, composition | 9.65 | 10.04 | 10.85 |
| 1-NN lookup, composition | 11.20 | 11.80 | 12.82 |
| Linear, engineered | 17.62 | 17.75 | 18.01 |
- Paired per split, TabPFN beats published-settings XGBoost on the same feature set in 25/25
  splits for random and grouped (mean RMSE gap 0.71-1.07 K) and on grouped_no_oxygen in 25/25
  (composition, -0.91 K) and 18/25 (engineered, -0.34 K).
- Leave-family-out: every model fails to extrapolate (held-out cuprates: RMSE 46-58 K for all
  models); TabPFN is not an exception. Coverage on the held-out family is Phase 3's question.

### Leakage checks before believing the gap (grouped split, all free)
- TabPFN's advantage over XGBoost is larger on test rows with no close training composition
  (split 0, L1 >= 0.05: 8.22 vs 9.60 K, engineered) than on rows with a near-duplicate
  (L1 < 0.01: 9.47 vs 9.91 K). Memorization or leakage would show the opposite.
- TabPFN predicts 6.1% of grouped test rows within 0.1 K (XGBoost 2.9%), but these are almost
  all low-Tc materials (13.3% of rows below 10 K vs 2.1% above): sharper predictions where Tc is
  a few kelvin, not leaked labels.

### Step 4 complete: tuned XGBoost added (2026-10-02)
Nested tuning (26 trials per split, 2.82 h of tuning compute over 156 split/feature jobs) ran
partly in a background task, which hit its time limit after 13 random splits, and was finished
by the author in a terminal (the script resumes from per-split records). Records:
`results/02_benchmark/xgb_tuning/`; predictions: `predictions/xgb_tuned/`.

RMSE (K), 25 splits, sqrt(mean MSE) [metrics.csv]:
| Model | Features | random | grouped | grouped_no_oxygen |
|---|---|---:|---:|---:|
| TabPFN-3.5 | composition | 8.70 | 8.97 | 9.95 |
| TabPFN-3.5 | engineered | 8.72 | 9.01 | 10.54 |
| XGBoost tuned | engineered | 9.34 | 9.73 | 11.04 |
| XGBoost tuned | composition | 9.55 | 9.90 | 10.88 |
| XGBoost published | engineered | 9.42 | 9.76 | 10.85 |
| XGBoost published | composition | 9.65 | 10.04 | 10.85 |
| 1-NN lookup | composition | 11.20 | 11.80 | 12.82 |
| Linear | engineered | 17.62 | 17.75 | 18.01 |

- Tuning barely helps XGBoost: at most ~0.1 K over the published settings, and worse on
  grouped_no_oxygen with engineered features (11.04 vs 10.85 K). The paper's settings were
  already near the optimum this search could find.
- Paired per split [paired.json], TabPFN vs tuned XGBoost on the same features: better on 25/25
  splits for random and grouped (mean gap -0.63 to -0.93 K), and on grouped_no_oxygen 25/25
  (composition, -0.94 K) and 19/25 (engineered, -0.53 K, SD 0.68: noisier).
- MAE gaps are larger than RMSE gaps (grouped, composition: TabPFN 4.96 vs tuned XGBoost 5.79 K).
- Grouped split by subgroup (RMSE, K; TabPFN composition vs best XGBoost): Tc < 10 K 4.53 vs
  5.02; 10-77 K 10.47 vs 11.00; Tc > 77 K 11.27 vs 12.74; cuprates 11.71 vs 12.72; iron-based
  7.18 vs 7.46 (TabPFN engineered 6.71); other 4.43 vs 4.77. TabPFN is ahead in every subgroup.
- Missing-oxygen rows (646 test rows per grouped split): highest error of any subgroup for every
  model (TabPFN 12.8-13.0 K, tuned XGBoost 14.1-14.7 K across split kinds), as expected from
  their wrong stoichiometry; reported separately as planned.
- Leave-family-out (RMSE, K): no model extrapolates. Held-out cuprates 46-59 K for every model
  (R² -1.2 to -2.5); iron-based 18-26 K; other 20-327 K (linear on composition explodes).
  TabPFN is best or near-best on two folds but still far off; Phase 3 asks whether its
  intervals widen on the held-out family.
- `make benchmark` rebuilds metrics.csv byte for byte from committed files, with no API calls.

## 2026-10-02 — Phase 2, step 5: figures, notebook, compute, Thinking probe

### Figures and notebook
- `experiments/02_figures.py` -> `results/02_benchmark/figures/`: `paired_difference_grouped.png`
  (per-split RMSE of TabPFN minus tuned XGBoost on the grouped split, both feature sets),
  `rmse_by_split_kind.png`, `grouped_subgroups.png`. Readable, not polished (polish is Phase 6).
- `notebooks/02_benchmark.ipynb` reads only `results/02_benchmark/`; `make benchmark` (part of
  `make reproduce`) verifies the TabPFN cache and rebuilds metrics and figures with no API calls.

### Compute for the same 156 split/feature jobs (for the write-up)
- Nested XGBoost tuning: 2.82 h of search time (sum of per-split search times in
  `results/02_benchmark/xgb_tuning/`, refits excluded; mean 65 s per job), on one 10-core
  Apple M5 laptop; 26 trials per job, run partly alongside the TabPFN requests.
- TabPFN-3.5, no tuning: the whole run took 1,301 s = 21.7 min wall-clock with 4 requests in
  flight (`experiments/02_tabpfn.py` log), and 0.33 h (19.8 min) of server-side fit+predict time
  summed over the 156 request records (mean 7.6 s per job). 1.56M tokens.
- So TabPFN without tuning was ~8x faster in wall-clock than the XGBoost tuning it beat, and
  needed no search at all; the comparison excludes XGBoost's published-settings fits (~5-8 s
  each) and TabPFN's network transfer.

### Thinking probe (`experiments/02_thinking.py --probe`, grouped split 0, composition)
- The account can run Thinking: the thinking fit (medium effort, RMSE metric,
  group_col = integer material ID) succeeded in ~99 s and was charged 10,000 tokens (live usage
  1,600,000 -> 1,610,000). The ledger over-counts it at 2 x 60,865 estimated (conservative).
- The predict was refused with HTTP 422: "Thinking mode currently supports only
  output_type='mean' for regression". It was not charged. So TabPFN-3.5-Thinking gives point
  predictions only: no quantiles, intervals, CRPS or exceedance probabilities.
- Open decision: run the add-on with `output_type="mean"` (point metrics only, RMSE/MAE vs
  standard TabPFN on the same splits)? The probe's 3-request cap is used up (2 logged), so this
  needs the plan changed before any request; the guard will refuse it otherwise.

## 2026-10-02 — Phase 2 add-on: TabPFN-3.5-Thinking (point predictions only)

Source: `results/02_benchmark/thinking/grouped/composition/split_00..09` (`experiments/02_thinking.py`,
two batches of 5 splits). Medium effort, `thinking_metric="rmse"`, `thinking_timeout_s=900`,
`group_col` = integer material ID (factorized scaled-composition key), composition features.
- Limitation for the write-up: Thinking regression returns only the predictive mean (HTTP 422
  for `output_type="main"`), so Thinking has no quantiles, intervals, CRPS or exceedance
  probabilities here. Everything distributional in this project is standard TabPFN-3.5.
- Cost and time: 252,160 tokens for 10 splits (live usage 1,610,000 -> 1,862,160): each split is a
  thinking fit of ~99-128 s plus a ~10-15 s mean prediction. The ledger's per-request estimate
  (60,865) over-counts this.
- Result, paired on grouped splits 0-9: Thinking is worse than standard TabPFN on all 10 splits,
  RMSE 9.69 vs 8.93 K (sqrt mean MSE; mean per-split gap +0.75 K, SD 0.15) and MAE 5.94 vs 4.95 K.
  Its RMSE is close to tuned XGBoost on the same splits (9.6-10.3 K per split).
- Cause not investigated, due to time (decision 2026-10-02). Report the numbers as they are.
  Untested candidates: the material-ID column may be used as a feature (the docs do not say it
  is dropped), or Thinking's internal validation, grouped by material, selects configurations
  that suit unseen materials less well.

## 2026-10-02 — Phase 3: accuracy of the quantile-grid summaries (2 requests)

Source: `results/03_uncertainty/quantile_vs_full.json` (`experiments/03_quantile_vs_full.py`):
`output_type="full"` on the first 400 test rows of grouped splits 0 and 1 (composition), same
training rows and seed as Phase 2, compared with Phase 2's committed 107-level summaries.
- Means agree to within 0.014 K, so a row's prediction does not depend on the other test rows.
- CRPS: median |exact - grid| 0.0009 K (max 0.07 K); mean CRPS 6.179 vs 6.180 K (split 0).
- Summary quantiles within 0.033 K. PIT within 0.005.
- P(Tc > 77 K): within 0.005, which is the grid's clamp: 72-79 of 400 rows have an exact
  probability below 0.5% and the grid reports 0.5%. So grid-based exceedance probabilities are
  exact down to 0.5% and floored there; Phase 4 can use the accepted 999-level grid for 0.1%.

## 2026-10-02 — Phase 3, TabPFN side (provisional; superseded by the Phase 3 results below)

Sources: `results/03_learning_curves/` and `results/03_uncertainty/` (`calibration.csv`, `lfo.csv`,
`p77_brier.csv`, `curves.csv`), written by `experiments/03_evaluate.py`; figures in
`results/03_uncertainty/figures/`; walkthrough in `notebooks/03_uncertainty.ipynb`.
Provisional until the XGBoost learning curves and uncertainty baselines (run on the home Linux
server "jarvis", see machines.json once copied back) are in.

- Learning curves (50 requests, 500,000 tokens; grouped splits 0-4, nested subsets, fixed test
  sets): TabPFN RMSE, composition features, 17.53 K at 100 rows, 15.34 at 300, 13.10 at 1k,
  11.08 at 3k, 9.33 at 10k, 8.94 at full size (~14.2k). Engineered features track it closely.
- Calibration, TabPFN, all three repeated splits: slightly conservative. Grouped split,
  composition: coverage 54.0 / 82.5 / 91.5 / 95.8% at nominal 50 / 80 / 90 / 95%; 95% width 27.3 K;
  CRPS 3.44 K. Random and grouped_no_oxygen behave the same (95% coverage 95.0-95.7%). Coverage
  stays at 95-96% in every Tc band and family and on missing-oxygen rows, where the 95% interval
  is wider (51.2 K vs 27.3 K overall): the model widens its intervals where its errors are larger.
- P(Tc > 77 K) is well calibrated: Brier 0.037 on the grouped split vs 0.150 for always
  predicting the base rate (0.044-0.050 on grouped_no_oxygen).
- Leave-family-out (composition features; engineered in lfo.csv): TabPFN partly knows when it is
  extrapolating, and not equally for every family.
  - Held-out "other": 95% interval widens 12x (113 K vs 9.4 K in-distribution); coverage
    94.6% (in-distribution 96.2%).
  - Held-out cuprates: widens 2x (89 K vs 43 K) but under-covers: 78.3% at 95%, 35.8% at 80%.
  - Held-out iron-based: barely widens (29.7 K vs 25.0 K) and under-covers badly: 59.8% at 95%,
    22.2% at 80%. Overconfident on the family whose Tc range overlaps the training families.
  - Per-row rank correlation between interval width and absolute error: 0.57 (other), 0.08
    (iron-based), 0.01 (cuprates).

## 2026-10-02 — Phase 3 results: learning curves and uncertainty (final)

Sources: `results/03_learning_curves/{curves,paired}.csv`, `results/03_uncertainty/{calibration,
lfo,p77_brier}.csv` from `experiments/03_evaluate.py --require-all`; figures in
`results/03_uncertainty/figures/`; `notebooks/03_uncertainty.ipynb`. All 212 XGBoost results were
run on the home Linux server "jarvis" (`results/03_learning_curves/machines.json`); TabPFN on the
Prior Labs API. XGBoost uncertainty baselines cover the grouped split and leave-family-out only.

### Learning curves (grouped splits 0-4, nested subsets, same test sets; RMSE in K)
| features | model | 100 | 300 | 1k | 3k | 10k | full |
|---|---|---:|---:|---:|---:|---:|---:|
| composition | TabPFN-3.5 | 17.53 | 15.34 | 13.10 | 11.08 | 9.33 | 8.94 |
| composition | XGBoost tuned | 19.64 | 16.61 | 14.14 | 12.36 | 10.35 | 9.82 |
| composition | XGBoost published | 18.17 | 15.49 | 13.68 | 11.96 | 10.40 | 10.01 |
| engineered | TabPFN-3.5 | 18.59 | 15.15 | 12.95 | 11.18 | 9.44 | 9.06 |
| engineered | XGBoost tuned | 20.32 | 16.94 | 14.19 | 12.00 | 10.09 | 9.70 |
- TabPFN beats tuned XGBoost at every size: on 5/5 splits in 10 of 12 size/feature cells and
  4/5 in the other two; -2.1 K at 100 rows (composition), -0.9 K at full size.
- Tuned XGBoost is worse than the published settings below ~3k rows: its inner validation set
  is 20% of the subset (20 rows at n = 100), so the search overfits the validation noise.
- Against the better XGBoost variant at each size (lower mean RMSE over the 5 splits; chosen on
  test results, so this favours XGBoost) [paired.csv, b = xgb_best]: published settings up to
  1k rows (3k for composition), tuned above. TabPFN still has lower mean RMSE at every size,
  winning 5/5 splits in 9 of 12 cells and 4/5 in 3, but the small-data gap shrinks: composition
  -0.65 K at 100 rows, -0.15 K at 300 (near a tie), -0.58 K at 1k, -0.88 K at 3k, -1.02 K at 10k,
  -0.89 K at full size; engineered -0.82, -1.16, -0.91, -0.81, -0.66, -0.64 K. So the claim is
  "better at every size by 0.6-1 K from 1k rows up", not "largest gain with little data".

### Calibration on the grouped split (composition features; engineered similar)
| | 50% | 80% | 90% | 95% cov. | 95% width | CRPS20 | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabPFN-3.5 | 54.0 | 82.5 | 91.5 | 95.8% | 27.3 K | 3.45 K | 8.97 K |
| XGBoost conformal | 49.5 | 79.6 | 89.6 | 94.8% | 44.0 K | 4.80 K | 10.21 K |
| XGBoost quantile regr. | 42.5 | 70.1 | 82.5 | 90.9% | 42.5 K | 4.54 K | 10.86 K |
- TabPFN is the only one that is calibrated and sharp at once: CRPS 25-28% lower than both
  XGBoost methods, 95% intervals ~38% narrower, and coverage at or slightly above nominal.
- Conformal is calibrated on average by construction but has one width (44 K) for every row:
  it over-covers easy rows (Tc < 10 K: 99.0%, family other: 99.5%) and under-covers hard ones
  (Tc > 77 K: 90.9%, cuprates: 90.4%, missing-oxygen rows: 87.4%).
- Quantile regression under-covers everywhere (86-94% at 95%).
- TabPFN keeps 95-96% coverage in every Tc band, family and on missing-oxygen rows by adapting
  its width (9.3 K for Tc < 10 K, 43 K for Tc > 77 K, 51 K on missing-oxygen rows).
- P(Tc > 77 K): Brier 0.037 (TabPFN) vs 0.050 (both XGBoost methods); base rate 0.150.

### Leave-family-out: do the intervals know when the model is extrapolating? (composition)
| held out | TabPFN 95% cov. (width) | QR 95% cov. (width) | conformal 95% cov. (width) |
|---|---|---|---|
| cuprate | 78.3% (89 K) | 27.8% (36 K) | 11.1% (17 K) |
| iron-based | 59.8% (30 K) | 75.3% (30 K) | 63.0% (42 K) |
| other | 94.6% (113 K) | 17.3% (36 K) | 80.7% (50 K) |
- Every method under-covers a held-out family. TabPFN widens its intervals where it extrapolates
  (2x for cuprates, 12x for "other") and keeps the highest coverage on those two folds; on
  held-out iron-based it barely widens and is beaten on coverage by quantile regression.
- Held-out CRPS20 (K): cuprate TabPFN 37.1 vs QR 48.0 vs conformal 42.4; iron-based 15.1 vs 14.6
  vs 14.3; other 14.1 vs 15.9 vs 14.4. TabPFN is clearly better only on held-out cuprates.
- Width tracks error per row (Spearman) only partly: TabPFN 0.57 (other), 0.08 (iron-based),
  0.01 (cuprate); QR 0.04-0.21; conformal undefined (constant width). Engineered features in
  lfo.csv.

## 2026-10-02 — Phase 4 setup: loop, gates, grid check

### Design (decided)
- Pools: main 15,147 materials (target Tc >= 119 K, 152 targets); hard 7,637 non-cuprates
  (target Tc >= 44.0 K, 79 targets, 76 iron-based). Composition features. Start: 50 random
  non-target materials per seed (same start for every method with that seed); 20 rounds of 10
  (200 "experiments"). Random search: ~2 targets expected in 200 picks.
- TabPFN acquisitions: EI over y* = min(best Tc so far, threshold); q90; greedy mean. Reported
  only: P(top 1%) and P(Tc > 77 K). Free baselines: greedy XGBoost (published settings, better
  than tuned below 1k rows) and random.
- Pilot: seeds 100-102, both scenarios, EI vs q90. Pre-registered rule: more targets found by
  round 20 summed over the 6 pilot runs wins; within max(1, 10% of the larger total) both go on.
  Main: seeds 0-9, winner(s) + greedy TabPFN.
- Gates between pilot and main (stop on any failure; none looks at which method wins): G1 all
  pilot runs completed 20 rounds without error; G2 simulated random search (1,000 replicates per
  scenario, through the same loop code) within 4 standard errors of its exact hypergeometric
  mean and P(>= 1 hit); G3 every batch only from unlabeled pool materials, start sets target-free,
  Tc/labels match the data; G4 the budget guard authorizes the main runs.
- Committed per run: per-round summaries (selected batch with EI, q90, mean, P(top 1%), P(77 K)
  and revealed Tc; pool totals; binned P(top 1%) vs outcome). Local only: per-material scores
  every round, raw grids for the first and last round.
- Note for the write-up (from the leave-family-out results): the hard scenario's targets are
  mostly iron-based, where TabPFN's held-out uncertainty was weakest. Unlike leave-family-out,
  the iron-based family is not hidden here (only the targets are). Report it whatever it shows.

### Grid check (2 requests; `results/04_discovery/grid_check.json`)
Main pool (15,097 unlabeled), fit on pilot seed 100's start set, tail-dense grid (146 levels)
vs uniform 999-level grid: EI max |diff| 0.005 K (Spearman 0.99999), q90 0.007 K (1.00000),
P(top 1%) 0.0002 (0.99999); the first batch of 10 is identical for all three. Request time 8.4 s
vs 34.7 s. Both grids end at the 0.9995 quantile, so both underestimate EI by the same amount
when y* is more than ~2.5 SD above the predictive mean (test_discovery).

### Dry runs
`experiments/04_discovery.py all --dry-run` (fake predictor, scratch outputs and ledger) ran the
whole pipeline; a planted repeated selection tripped G3 and blocked the main runs with 0
requests; a run cut after round 1 resumed to an identical result charging only the missing
rounds; a rerun with everything complete charged nothing.

## 2026-10-03 — Phase 4 results: simulated discovery

Sources: `results/04_discovery/` (run records under `runs/`, `pilot_decision.json`, `gates.json`,
`summary.csv`, `curves.csv`, `paired.csv`, `novelty.csv`, `families.csv`, `reliability.csv`,
`batch_calibration.csv`, `timing.json`), from `experiments/04_discovery.py all --live` (one
overnight command, 62.5 min) and `experiments/04_evaluate.py`; figures in
`results/04_discovery/figures/`; `notebooks/04_discovery.ipynb`.

### Run
- Pilot (seeds 100-102): EI found 441 targets in total over the 6 pilot runs, q90 378. The
  pre-registered rule chose EI (gap 63 > 10% of 441). Per run, EI 112 / 90 / 96 (main) and
  59 / 62 / 22 (hard); q90 86 / 89 / 84 and 66 / 1 / 52.
- Gates all passed: G1 pilot complete; G2 random simulation 1.98 vs exact 2.01 targets (main) and
  2.18 vs 2.08 (hard), P(>= 1 hit) 0.871 vs 0.870 and 0.886 vs 0.880; G3 every batch valid;
  G4 authorized 400 requests per scenario (EI + greedy TabPFN). Final integrity check: all valid.
- Cost: 1,040 billed requests = 10,400,000 tokens (live usage 2,402,160 -> 12,802,160). The
  ledger shows 4 extra charges: retries on 4 parallel hard/greedy_tabpfn runs within 21 s; the
  first attempts were not billed (live usage = 1,040 x 10k) and all retries succeeded. Their
  error text was not recorded; the runner now logs it. Median request 11.5 s (main pool),
  7.6 s (hard).

### Targets found after 200 experiments (main runs, seeds 0-9; 95% t-interval over seeds)
| | main (152 targets) | hard (79 targets) |
|---|---|---|
| TabPFN EI | 79.6 (65.2-94.0) | 53.9 (42.4-65.5) |
| TabPFN greedy (mean) | 42.5 (16.2-68.8) | 26.7 (4.7-48.7) |
| XGBoost greedy (published settings) | 35.6 (11.5-59.7) | 33.6 (10.0-57.2) |
| random search | 2.1 (exact expectation 2.01) | 2.3 (exact 2.08) |
- Enrichment over random at 200 experiments: EI 39.5x (main), 25.9x (hard).
- Paired by seed: see the next section. EI minus random: +77.5 (main) and +51.6 (hard), 10/10
  seeds each.
- Greedy XGBoost is fastest early (15.3 vs 10.3 targets after 50 experiments, main; 10.0 vs 5.7
  hard) and has the shortest median time to a first hit (2 experiments main, 27 hard, vs EI 20
  and 37), but 3 of its 10 runs found no target at all in either scenario. Every EI run found
  targets. EI is ahead of greedy XGBoost from 80 (main) and 120 (hard) experiments onwards [curves.csv].
- Hard scenario, reported as it came out (its targets are iron-based, where TabPFN's
  leave-family-out uncertainty was weakest; here the family is not hidden, only its targets):
  EI finds 538 iron-based targets and 1 other over 10 seeds and leads every method, but its
  margin over greedy XGBoost is the least certain result (6/10 seeds), and P(top 1%) is
  miscalibrated there (below).

### Is the uncertainty doing the work? EI vs greedy, paired by seed
Sources: `paired.csv`, `paired_seeds.csv`, `stalls.csv`, `paired_by_stall.csv`,
`ablation_check.json`; figure `figures/per_seed.png`.
- The clean ablation is EI vs greedy TabPFN: same model, same 50-material start set and the same
  first fit. Round-1 training sets are identical on 10/10 seeds in both scenarios, and the 28
  materials both methods picked in round 1 got identical predicted means. Only the selection
  rule differs: the full predictive distribution (EI) or its mean (greedy). Greedy XGBoost is
  the cross-model comparison.
- After 200 experiments, EI minus greedy TabPFN: +37.1 targets (main, 95% interval +6.6 to +67.6,
  EI better on 9/10 seeds) and +27.2 (hard, +7.4 to +47.0, 7/10). EI minus greedy XGBoost:
  +44.0 (main, +17.8 to +70.2, 9/10) and +20.3 (hard, +0.5 to +40.1; 6 better, 3 worse, 1 tie).
- The gap builds late. After 50 experiments, EI minus greedy TabPFN / greedy XGBoost is -0.4 /
  -5.0 (main) and 0.0 / -4.3 (hard); after 100, +4.6 / +3.2 and +5.9 / -3.1. All of these
  intervals include zero.
- Most of the gap comes from greedy runs that stall. Post hoc, defined after seeing the per-seed
  results: a run stalls if it finds at most 5 targets in 200 experiments (random expects ~2).
  Both greedy methods stall on the same start sets: main seeds 0, 4, 8 (0 targets each) and hard
  seeds 0-3 (at most 2). EI never stalls (fewest: 48 main, 23 hard). The cut-off does not drive
  this: stalled runs found at most 2 targets, every other run at least 10. On those start sets
  EI, with the same model and first fit as greedy TabPFN, found 74 (main) and 41 (hard) targets
  on average. Why greedy stalls there is a Phase 5 question.
- Where the greedy run did not stall, the gap is smaller. Vs greedy TabPFN: +21.3 (main, EI better
  on 6/7 seeds; greedy TabPFN has the single best run, 111 vs EI's 52 on seed 1) and +18.3
  (hard, 3 better, 3 worse). Vs greedy XGBoost: +31.1 (main, 6/7) and +6.8 (hard, 2 better,
  3 worse, 1 tie).
- Reading for the write-up: using the uncertainty mainly buys robustness (no run stalls), not a
  uniformly faster search. On the hard pool, a greedy run that doesn't stall is about as good
  as EI.

### Caveats that stay in the write-up
- Greedy XGBoost is faster to the first hit and ahead for the first 80 (main) / 120 (hard)
  experiments.
- The hard pool's margin over greedy XGBoost is narrow (6/10 seeds; interval down to +0.5), and
  near zero on seeds where greedy XGBoost doesn't stall.
- TabPFN's P(top 1%) is overconfident on the hard pool (below).
- The stall cut-off is post hoc (the gap between 2 and 10 targets makes the split
  insensitive to it); 10 seeds per method.
- Simulated discovery: every pool material is a known superconductor (the data has no
  non-superconductors), "measuring" reveals a recorded Tc, and the targets are known
  materials. It tests how well each method ranks candidates, not whether new materials exist.

### Checks before believing the size of the effect
- No label leakage path: each round fits only on measured materials (start set + earlier picks);
  start sets contain no target (G3); random search matches its exact law (G2).
- Novelty of the targets found (L1 over element fractions to the nearest already measured
  material; YBa2Cu3O7 vs O6.9 = 0.007): main, EI 9.3% within 0.01, 49.5% within 0.05, 34.0%
  beyond 0.1, 11.2% oxygen variants of a measured material; greedy XGBoost 10.4 / 54.2 / 30.1 /
  7.9%. Hard, EI 5.6 / 27.5 / 55.5 / 12.1%; greedy XGBoost 7.4 / 33.9 / 40.5 / 15.5%. So about
  half of the main-pool finds are close relatives of something already measured, for every
  method alike; EI's finds are not more derivative than the baselines', and in the hard pool
  they are further from anything measured (median 0.122 vs 0.063).

### Reliability of TabPFN's probabilities during discovery (EI runs)
- Batch level: expected (sum of P(top 1%)) vs found targets 761.6 vs 796 (main), 575.3 vs 539
  (hard); expected vs found Tc > 77 K among measured materials 1,567.1 vs 1,568 (main).
- Pool level (every unlabeled material, every round): calibrated on the main pool (e.g. 0.20-0.50
  bin: predicted 0.302, observed 0.283; >= 0.5: 0.652 vs 0.653). On the hard pool overconfident
  above 0.02 (0.20-0.50: 0.302 vs 0.169; >= 0.5: 0.607 vs 0.442) and underconfident below 0.001
  (0.0006 vs 0.0027), consistent with the iron-based leave-family-out result.

## 2026-10-03 — Phase 5: why analysis (no API requests)

### GP-EI discovery baseline (decided before running; reported as it came out)
Question: does EI need TabPFN's predictive distribution, or does any uncertainty-aware model
do? Configuration fixed in the Phase 5 plan before any GP run (`models.gp_predict`): scikit-learn
GP, amplitude x Matern 5/2 with one length scale + white noise, normalised Tc, 3 optimizer
restarts, element-fraction features, default hyperparameter bounds; closed-form EI with the same
y*; seeds 0-9, same start sets, 20 rounds of 10, both scenarios. (The amplitude is the standard
output scale of a Matern kernel; the plan did not spell it out, and it was set before any run.)
Not part of the pre-registered Phase 4 design. `experiments/05_gp_discovery.py` (`make
gp-discovery`, 48 s for all 20 runs); records in `results/04_discovery/runs/main/*/gp_ei/`, all
pass the integrity checks; evaluated with the Phase 4 runs [summary.csv, paired.csv, stalls.csv,
paired_by_stall.csv, batch_calibration.csv, reliability.csv, novelty.csv, gp_fits.json].
- Targets found after 200 experiments: GP EI 28.8 (95% interval 11.7-45.9) main, 20.2 (1.8-38.6)
  hard, vs TabPFN EI 79.6 and 53.9, greedy TabPFN 42.5 and 26.7. Median experiments to a first
  hit 28 (main) and 76 (hard); 1 and 3 runs without any hit.
- Paired by seed, TabPFN EI minus GP EI: +50.8 (main, +27.1 to +74.5, TabPFN EI better on 9/10
  seeds) and +33.7 (hard, +16.7 to +50.7, 10/10). After 50 experiments +5.5 (main, 9 better,
  1 worse) and +2.0 (hard); after 100, +15.2 and +12.2.
- GP EI stalls too: main seeds 4, 7, 8 (at most 3 targets), hard seeds 0, 1, 2, 3, 6, 7 (at most
  2); its other runs found at least 19 (main) and 44 (hard). On the seeds where it did not
  stall, TabPFN EI is still ahead: +36.9 (main, 6/7) and +12.3 (hard, 4/4).
- Its probabilities are not usable as probabilities. Pool level, GP P(top 1%) 0.10-0.20:
  observed 0.0098 (main) and 0.0086 (hard); 0.20-0.50: 0.016 (main); the observed rate is
  roughly flat at 1-2% from predicted 0.01 to 0.2. TabPFN's in the same bins: 0.130 (main,
  predicted 0.137). Batch level, expected vs found targets: 370.0 vs 288 (main), 197.7 vs 202
  (hard).
- The fits are not degenerate: 2 and 4 convergence warnings in 200 fits per scenario; fitted
  noise median 0.093 (main) and 0.156 (hard) of the normalised variance, length-scale medians
  0.38 and 0.48 on element fractions [gp_fits.json].
- Reading: with the same acquisition rule, swapping TabPFN's distribution for this GP's loses
  most of the gain, and GP EI is below even greedy TabPFN on average. So the gain needs EI and
  TabPFN's distribution together.
- Caveat for the write-up: one GP configuration, fixed in advance and not tuned (single length
  scale over 86 sparse element fractions). A GP with per-feature length scales, other
  features or a warped output might do better; this tests a standard default, not GPs in
  general.

### Where TabPFN beats tuned XGBoost (grouped split, 25 repeats)
Sources: `experiments/05_where_wins.py` -> `results/05_why/where_by_distance.csv`,
`where_by_subgroup.csv`, `worst_materials.csv`, `worst_summary.json`. Uses the committed Phase 2
predictions; distance = L1 over element fractions from each test row to the nearest training
material (all 25 splits; matches Phase 1's split 0).
- Distances: 40.7% of test rows are within 0.01 of a training material (mostly oxygen
  variants), 31.7% at 0.01-0.05, 10.5% at 0.05-0.1, 7.8% at 0.1-0.2, 9.3% at 0.2 or more.
- TabPFN is ahead in every distance bin on both feature sets. Composition features, per-split
  RMSE difference: -0.72 K (< 0.01), -1.30 (0.01-0.05), -0.88 (0.05-0.1), -0.77 (0.1-0.2,
  TabPFN better on 22/25 splits), -0.81 (>= 0.2, 24/25); 25/25 in the three nearest bins.
  The nearest bin holds 40.7% of the rows but 30.7% of the MSE advantage (engineered:
  25.5%), so the advantage is not a near-duplicate effect.
- Within families, the gap grows with distance: cuprates -0.81 K (< 0.01) to -2.68 K (>= 0.2);
  iron-based -0.14 K (< 0.01, 19/25 splits) to -2.10 K (0.1-0.2). The one cell without a
  TabPFN edge: "other" materials at 0.1-0.2 (+0.13 K, interval -0.37 to +0.62, TabPFN better
  on 14/25 splits).
- By subgroup (composition): ahead in every family (cuprates -1.25 K, iron-based -0.71, other
  -0.36 with 21/25 splits), every Tc band (-0.65 / -0.88 / -1.49 K for < 10, 10-77, >= 77 K)
  and on missing-oxygen rows (-1.45 K). Cuprates carry 87.5% of the MSE advantage (50% of
  rows) because their errors are largest. Per row, TabPFN is closer more often on "other"
  materials (70.4% of rows) than on cuprates (56.9%).
- Worst materials (mean absolute error over the splits that test each material): 6 of the 25
  worst are among the 17 entries Phase 1 flagged as suspect (0.1% of all materials). They are
  H2S (high pressure), two unconfirmed tungsten bronzes, a C60 compound and two formulas that
  lost their Cu (Y1Ba23O at 90.4 K, Bi2Sr2Ca2O at 109 K). The worst 1% (151 materials) are
  90% cuprates (49.5% overall) and 26.5% missing-oxygen rows (9.6% overall). XGBoost is also
  off by more than 30 K on 84% of them, so these are mostly label and representation problems
  both models share.

### Why greedy search stalls (descriptive, post hoc)
Sources: `experiments/05_stalls.py` -> `results/05_why/stall_runs.csv`,
`stall_trajectories.csv`, `stall_round1.csv`, `stall_summary.json`, `figures/stalls.png`.
- What the targets are: 93.4% of main-pool targets contain Hg or Tl; 96.2% of hard-pool targets
  are iron-based.
- Both greedy methods stalled on main seeds 0, 4, 8 and hard seeds 0-3. There they plateau just
  below the threshold: best Tc at most 99 K after 200 experiments (main, threshold 119 K) and at
  most 47.2 K (hard, threshold 44 K). By round 5 they were at about 96 K and 41 K and barely
  moved afterwards.
- Their picks are large families of near-threshold non-targets. Main, most-picked systems:
  Ba-Cu-Y (8.4% of picks), Ba-Cu-Hg (7.7%), Ba-Cu-La-Y, Ba-Cu-Pr-Y, Ba-Cu-Sr-Y (YBCO type
  and Hg-1201, median Tc of picks 89.7 / 88.9 K). Hard: B-C-Mg (13.4%), Al-B-Mg (12.1%),
  B-Li-Mg, B-Cu-Mg (doped MgB2, median 34.6 / 33.8 K). Only 12% (main, Hg or Tl) and 21-26%
  (hard, iron-based) of their picks are in the target chemistry.
- The predictions behind these picks were accurate. Greedy TabPFN on the stall seeds
  predicted 86.6 K and measured 86.8 K on average (main), 32.9 vs 31.6 K (hard). The model was
  right that these materials are good; greedy search just kept measuring them.
- TabPFN EI on the same start sets put 73% of its picks in the target chemistry (main: Tl-Ba-Ca-Cu,
  Hg-Ba-Ca-Cu systems) and 50% (hard: Fe-As with Nd or Sm), found 74 and 41 targets on average,
  with a mean best Tc of 128.8 K after round 5 (main) and 50.6 K after round 10 (hard).
- Round 1, same fit, all 10 seeds: EI's first batch has a lower predicted mean than greedy's
  (-21.9 K main, -6.7 K hard) and a wider upper tail (q90 - mean: +17.0 K, +4.4 K). It
  realises lower Tc (-42.7 K, -12.7 K) and, on the main pool, 1.4 fewer targets in that
  round. EI pays for exploring at the start, which is why its curve starts slower.
- Start sets: on the main pool the three stall seeds have the weakest best start material
  (mean 89.9 K vs 108.5 K for the other seeds). On the hard pool there is no such difference
  (37.8 vs 35.6 K); no simple start-set feature predicts the hard-pool stalls.
- GP EI stalls the same way on the hard pool (6 seeds; 10% of picks iron-based, best 43.8 K
  on average).
- Reading: greedy search is held back by correct predictions, not wrong ones. A large family
  of materials just below the threshold looks best on the mean, and nothing in the mean says
  that a less certain chemistry might be higher. EI on TabPFN's distribution scores the
  upper tail and goes there. GP EI has the same rule, but its distribution is not reliable
  enough on the hard pool (see the GP section).

### Do TabPFN's intervals widen away from the training data?
Sources: `experiments/05_intervals.py` -> `results/05_why/interval_vs_distance.csv`,
`interval_vs_neighbour_spread.csv`, `width_drivers.csv`, `lfo_neighbours.csv`,
`figures/interval_width_drivers.png`. Composition features; TabPFN and both XGBoost uncertainty
baselines. "Neighbour Tc spread" = SD of the median Tc of the 10 nearest training materials
(L1 over element fractions).
- Grouped split: within each family, TabPFN's 80% interval widens with distance. Cuprates go
  from 20.9 K (< 0.01) to 51.5 K (>= 0.2), iron-based from 13.3 to 31.5 K, other from 4.1 to
  6.7 K. Coverage stays 0.77-0.86 (80%) and 0.945-0.972 (95%) in every family x distance
  cell. Pooled over families the width falls with distance (16.7 to 10.8 K), only because
  distant test materials are mostly low-Tc "other" materials.
- What the width tracks on the grouped split. Spearman correlation of the 80% width (per split,
  mean over 25):
  - TabPFN: 0.83 with the neighbour Tc spread, 0.74 with |error|, -0.19 with distance.
  - XGBoost quantile regression: 0.83, 0.64 and -0.29 (conformal widths are constant).

  Both models' widths mostly reflect how much Tc varies among similar materials. TabPFN's
  width tracks its own errors more closely.
- Leave-family-out: held-out materials are far from everything in training (median distance
  0.62 for cuprates, 0.80 iron-based, 1.45 other; at most 0.6% within 0.05). Within a
  held-out family, the width barely tracks anything: Spearman with the neighbour spread 0.42
  (cuprates), 0.09 (iron-based), 0.52 (other); with |error| 0.02, -0.03 and 0.67.
- The iron-based case. In Phase 3, TabPFN's intervals got narrower when this family was held
  out, with 22% coverage at 80%. The nearest training materials are low-Tc analogues from the
  "other" family (99.9%): As-Ba-Pt (16% of rows), As-Ca-Pd (9%), As-F-La-Ni-O (8%), Fe-O-P-Sm
  (7%), Fe-S-Te (7%). Their Tc is low and uniform (10 nearest: median mean 4.2 K, median SD
  2.0 K). TabPFN predicts a median of 5.4 K (true median 20.0 K) with a median 80% width of
  8.5 K. Neighbours of held-out cuprates and of held-out "other" materials vary far more (SD
  14.2 and 16.8 K), giving wide intervals (median 41.8 and 77.6 K) that still under-cover
  (36% and 52%).
- The Phase 5 plan's hypothesis was that held-out iron-based materials sit close to training
  materials. That is not supported: they are far. The explanation is that their nearest
  analogues all have similar, low Tc.
- Reading for the write-up: TabPFN's interval width reflects how much Tc varies among similar
  training materials.
  - Within the training distribution this is a good signal: coverage holds at every distance,
    and width grows with distance within each family.
  - It is not a novelty detector. When a whole family is unseen, the width comes from whichever
    family is nearest, and a family whose nearest analogues are uniformly low-Tc gets
    confidently wrong intervals.
  - The hard-pool overconfidence in Phase 4 is consistent with this, although there the
    iron-based family is not hidden.

### Is Hamidieh's top feature a cuprate detector?
Sources: `experiments/05_cuprate_detector.py` (`make why-xgb`, 76 s, local XGBoost fits) ->
`results/05_why/cuprate_detector.json`, `feature_gain.csv`.
- On its own, `range_ThermalConductivity` separates cuprates almost perfectly: ROC AUC 0.993.
  98.9% of cuprate rows have the value 399.973 within 0.001 (Cu minus O; two near-identical
  values, 399.97342 and 399.97417; Phase 1's 98.5% rounded to 3 decimals and counted only the
  first), and no non-cuprate row has it.
- Published XGBoost refit on grouped splits 0-4, engineered features. The refits reproduce the
  committed Phase 2 predictions exactly (max difference 0.0).
  - The feature carries 23.0% of the total gain on average; the paper reports about 30% on its
    own fit.
  - With an explicit `is_cuprate` column added, `is_cuprate` takes 27.7% of the gain,
    `range_ThermalConductivity` falls to 10.5%, and test RMSE is unchanged (9.77 to 9.75 K).
- Reading: the hypothesis holds. About half of the paper's most important feature is family
  membership, which the element fractions already encode.

## 2026-10-06 — Phase 6: demo, README, CI

### Demo (`demo/predict.py`, `demo/showcase.py`; 20 requests, `p6_demo`)
- Formula -> Tc: TabPFN-3.5 fit on all 21,263 rows with composition features, seed 0, the
  107-level grid. If the material (same scaled composition, any spelling) is in the data, all
  its rows are held out of training, so each known material gets its own request. Oxygen
  variants and other relatives stay in (as in the grouped split); the demo prints the nearest
  training material, its distance and Tc, because Phase 5 showed that the interval width is
  not a novelty detector. Warns when the nearest material is 0.2 or more away (L1).
- Cache: one JSON per formula in `results/06_demo/predictions/` with the full quantile grid
  and a fingerprint of the inputs (same `models.fingerprint` as the benchmarks); written
  straight after the request. Without `--live` the demo only reads the cache. Dry-run with a
  fake predictor in `tests/test_demo.py` before the live run.
- Live usage reading before the run (`results/api_usage.jsonl`, 2026-10-06T18:35 UTC): the
  October pool limit is now 60M tokens (raised), 13.80M used. The usage rose about 1.0M since
  the Phase 4 reading (12.80M) without entries in this project's ledger; not investigated.
- One live request first (MgB2, 21,216 training rows, 11.5 s), then the other 19 showcase
  formulas (3.2 min). Ledger: 20 requests.
- Showcase [`results/06_demo/showcase.csv`, `figures/showcase.png`]: 16 textbook
  superconductors from the data, each held out; every recorded median Tc lies inside its 95%
  interval and 15 of 16 inside the 80% interval (e.g. HgBa2Ca2Cu3O8: median 131.5 K vs
  recorded 131.0 K; MgB2 37.6 vs 38.4 K; the exception is Tl2Ba2Ca2Cu3O10, 111.5 vs 119.5 K).
  16 materials, so this is an illustration, not an evaluation. 4 materials not in the data
  (CsV3Sb5, Nd0.8Sr0.2NiO2, La3Ni2O7, LaH10): no ground truth, shown for the distance warning.
  LaH10 (nearest training material H4Si1 at L1 0.40) gets 95% interval -0.1 to 287 K and
  P(Tc > 77 K) 19.9%; the data has no pressure, and the model says it does not know.

### README and reproducibility
- `experiments/06_readme.py` renders every README table from a committed results file between
  `<!-- readme:NAME -->` markers (`make readme`, part of `make reproduce`);
  `tests/test_readme.py` fails if the README and `results/` disagree. Prose numbers were
  checked against their files when written (sources named next to them).
- Fresh clone on macOS (scratch directory, `make setup && make reproduce`): 3.5 min, 84 tests
  pass, `git status` clean afterwards, so every result, figure and notebook is byte-identical.
- CI (`.github/workflows/reproduce.yml`): `make reproduce` on a fresh Ubuntu clone, then
  `.github/check_unchanged.py` fails on any changed or new file except PNGs and the images
  embedded in executed notebooks (checked locally with a planted notebook-image change, which
  passes, and a planted metrics change, which fails).

### Cross-platform determinism (found by the first CI run, 2026-10-06)
- On Ubuntu (x86) `make reproduce` and all tests passed, but 4 CSVs differed from the macOS
  (ARM) commit. Causes: `duplicates.csv` sorted by `n_rows` with an unstable sort (ties came
  out in another order), and the 10-nearest-neighbour Tc spread (`src/benchmark.py`) used
  `np.argpartition`, which picks different neighbours among tied distances on x86 and ARM.
  Both now use a stable sort, so ties resolve by index on every platform. The notebook check
  also ignores the rendered image size metadata (fonts).
- Effect on reported numbers (kNN neighbour sets changed only where distances tie): the
  grouped-split Spearman values are unchanged at 2 decimals (TabPFN 0.83, QR 0.83); in
  leave-family-out, TabPFN width vs neighbour spread for held-out iron-based 0.01 -> 0.09
  (cuprates 0.42, other 0.52 unchanged); iron-based neighbours' median mean / SD 4.4 / 2.1 K
  -> 4.2 / 2.0 K; neighbour SD for held-out cuprates / other 16.5 / 17.4 K -> 14.2 / 16.8 K.
  Widths, coverage and every other result are unchanged. The Phase 5 text above is updated.
