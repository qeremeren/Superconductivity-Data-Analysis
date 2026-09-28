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
- Benchmark and discovery predictions use `output_type="quantiles"` with a dense grid in one
  request per fit, not `"full"` (400-row cap). P(Tc > 77 K) comes from interpolating the
  quantile function and CRPS from the quantile-score integral over the grid; check the
  approximation error against `"full"` on a small subset in Phase 3.
- Cache per-row quantile grids (small); full logits only for a handful of demo rows
  (100 rows of float64 logits compress to ~1.5 MB).
- Plan API use by request count: 25 random splits x 2 feature sets = 50 requests; learning
  curves = sizes x seeds; discovery loop = rounds x seeds x acquisition functions.
