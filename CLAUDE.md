# CLAUDE.md — Superconductor Tc Discovery with TabPFN-3.5

## What this project is
Hackathon entry for the Prior Labs TabPFN-3.5 Hackathon (deadline: 6 Oct 2026, 23:59 CEST = 14:59 Pacific).
We use TabPFN-3.5 to (1) predict superconducting critical temperature (Tc) from chemical composition,
(2) benchmark it honestly against XGBoost, including the published Hamidieh (2018) baseline, and
(3) run a simulated materials-discovery loop that uses TabPFN's predictive distribution to find
superconductors above 77 K (liquid-nitrogen temperature) with as few "experiments" as possible.

Judging weights: TabPFN-3.5 showcase 50% · creativity/originality 30% · technical quality & reproducibility 20%.
Every design decision should serve one of these.

## Non-negotiable rules
- **No fabricated or hand-typed results.** Every number, table and figure in the README must be produced by code in this repo and traceable to a file in `results/`.
- **Report results honestly.** If XGBoost wins on some metric, say so and explain why. Do not write conclusions before the experiments exist.
- **Fair comparison.** XGBoost gets (a) the published Hamidieh settings and (b) a real tuning budget. Same splits, same seeds, same metrics for every model.
- **No data leakage.** Duplicate formulas must never appear in both train and test. Preprocessing is fit on train only.
- **Reproducible.** Fixed seeds, pinned dependencies, one command to reproduce all figures. All TabPFN API outputs are cached to disk and committed so judges can regenerate every figure without an API key; a flag re-runs the live calls.
- **License.** Repo is Apache License 2.0. Data is downloaded by script from UCI with attribution (check the dataset's license on its UCI page and cite it). Do not commit model weights.
- **Before writing any TabPFN code**, read the current docs (docs.priorlabs.ai, the TabPFN-3.5 changelog, and the README of github.com/PriorLabs/tabpfn-client and github.com/PriorLabs/TabPFN). Do not guess API signatures.

## Data
- UCI Machine Learning Repository, "Superconductivty Data" (UCI spells it this way; dataset id 464 — verify).
- `train.csv`: 21,263 rows, 81 engineered features + `critical_temp`.
- `unique_m.csv`: element-composition columns + `critical_temp` + `material` (formula string). Same row order as `train.csv` — verify before relying on it.
- Source paper: K. Hamidieh, "A data-driven statistical model for predicting the critical temperature of a superconductor," Computational Materials Science, 2018 (verify citation details).

## What the original paper did (Hamidieh 2018, arXiv:1803.10260)
- Data: NIMS SuperCon, pulled July 2017, 31,611 rows cleaned to 21,263. Rows with Tc = 0 or missing were dropped, so every row IS a superconductor (the model never learns "not a superconductor").
- Oxygen content like O7-X was rounded to O7. This adds label noise, especially for cuprates, where oxygen content strongly changes Tc.
- Only exact duplicate rows were removed. The same formula can still appear many times with different Tc (their own example: La1.8Ba0.2CuO4 appears ~13 times, Tc from ~9 K to 38 K).
- Protocol: random 2/3 train / 1/3 test, repeated 25 times, RMSE from the mean MSE.
- XGBoost: grid search on ONE random split (eta, colsample, max depth 15–25, min child weight, 750 trees), then evaluated on 25 new random splits. Best: eta 0.02, max_depth 16, min_child_weight 1, colsample 0.5, subsample 0.5, 374 trees → RMSE ≈ 9.5 K, R² ≈ 0.92. Multiple regression baseline: RMSE ≈ 17.6 K, R² ≈ 0.74.
- Top feature by gain: range_ThermalConductivity (~0.30 of total gain alone). Hypothesis to test: it acts largely as a cuprate detector (Cu has very high thermal conductivity, O very low).
- No per-material uncertainty: the paper gives one global ±9.5 K for every prediction.
- The paper suggests the model could help researchers narrow the search for high-Tc materials, but never tests that. Our discovery loop tests exactly this.
- Stated limitations: no pressure or crystal-structure information; poor predictions for materials unlike anything in the training data.
- Their R code and data: github.com/khamidieh/predict_tc

## What we do differently (and what we deliberately do NOT do)
Do: replicate their exact protocol for both models; add a formula-grouped split to measure how much duplicate leakage inflates the headline RMSE; per-material predictive distributions and calibration; learning curves; leave-family-out extrapolation; the discovery loop.
Do NOT: add crystal structure, pressure, new data sources, or a superconductor/non-superconductor classifier. Scope stays on this dataset.

## Evaluation protocol
Splits (all defined once in `src/splits.py`, saved to `data/splits/`):
1. `random` — the paper's exact protocol (2/3 train / 1/3 test, 25 repeats, RMSE = sqrt(mean of MSEs)), run for both XGBoost and TabPFN. Used only for apples-to-apples comparison with the published numbers.
2. `grouped` — grouped by scaled composition (element fractions summing to 1) so duplicates never cross train/test; 2/3 train / 1/3 test of rows, 25 repeats, matching `random` so the only difference is grouping. **Primary benchmark.**
3. `leave-family-out` — hold out whole families (cuprates: Cu>0 and O>0; iron-based: Fe>0 and (As>0 or Se>0); everything else). Tests extrapolation.
4. `grouped_no_oxygen` — sensitivity check: like `grouped`, but groups by composition with oxygen dropped, so all oxygen variants of a material stay on one side.

Metrics:
- Point: RMSE, MAE, R², overall and per Tc band (<10 K, 10–77 K, >77 K), and separately on rows whose formula gives no oxygen amount (encoded as O = 1; kept in the data, reported as a limitation).
- Distributional (TabPFN vs. XGBoost with quantile regression or conformal intervals): CRPS where available, 80%/95% interval coverage and width, calibration plot.
- Learning curves: train sizes 100, 300, 1k, 3k, 10k, full. Repeat each size with multiple seeds and report mean ± std.

## Discovery loop (the headline experiment)
- Pool-based active learning over deduplicated materials: one row per scaled composition, Tc = median over its duplicates. The 17 rule-flagged materials in `data/discovery_exclusions.csv` are left out of the pools (benchmarks keep them).
- Target: the top 1% of Tc in the pool (main scenario), and the top 1% of Tc among non-cuprates with a non-cuprate-only pool (hard scenario). Tc > 77 K is too common in the pool to separate methods (Phase 1 numbers in NOTES.md).
- Start: small random labeled set with no target material.
- Each round: fit on labeled set, predict the pool, pick a batch by acquisition function, reveal true Tc.
- Main acquisitions (TabPFN): expected improvement with reference min(current best, top-1% threshold), and the upper-quantile (q90) score; the pilot decides which one carries the main runs. Comparisons: greedy mean (TabPFN), greedy mean (XGBoost), random.
- P(Tc > 77 K) is reported as a metric, not used as an acquisition: a 99-level quantile grid cannot resolve it below 1%.
- Every billed TabPFN request goes through `src/budget.py` (`authorize()` / `RunBudget.charge()`); a run projected to exceed its cap is refused.
- Metric: number of distinct target materials found vs. experiments spent; rounds to the first hit; compared against random search's expected tries-to-hit.
- Multiple seeds, plotted with confidence bands. Budget API calls before running (batches reduce call count).

## Phase 6 requirements (keep every phase compatible with these from now on)
- `make reproduce` works from a fresh clone with **no API key**: downloads the data, rebuilds the splits, and regenerates every result, table, figure and notebook from the committed caches. It never calls the TabPFN API; a missing cache file is an error, never a silent live call. Each phase extends the target as it lands.
- The budget guard must not count the author's spend against anyone else's re-run. The guard reads only the per-clone ledger `.budget/ledger.jsonl` (gitignored) and the caller's own live API usage. The committed `results/api_ledger.jsonl` is the author's provenance record: appended only when `TABPFN_RECORD_SPEND=1` and never read by the guard.
- README quickstart includes `brew install libomp` for macOS (XGBoost's OpenMP runtime).
- The data download verifies a checksum (SHA-256 of each CSV, pinned in `src/data.py`).
- A GitHub Actions workflow runs `make reproduce` on a fresh Ubuntu clone and checks that results are unchanged (figures excepted: fonts differ by OS).

## Repo layout
```
data/            download script output (gitignored raw), splits/
src/             data.py, splits.py, models.py, metrics.py, discovery.py
experiments/     one script per experiment, each writes to results/
results/         cached predictions, metrics JSON/CSV, figures
notebooks/       01_eda, 02_benchmark, 03_uncertainty, 04_discovery (read from results/, no heavy compute)
demo/            formula -> Tc prediction with interval and P(Tc > 77 K)
tests/           splits have no leakage, metrics are correct, seeds reproduce
```

## How to work
- Plan first, then implement one phase at a time; show me the plan before large changes.
- Small commits with clear messages.
- Run the tests after changes to splits, metrics or data loading.
- When a result looks surprisingly good, check for leakage before celebrating.
- Keep a running `NOTES.md` of decisions and findings; the README is written from it at the end.

## Phases
0. Repo, license, environment, data download script, CLAUDE.md, tests skeleton.
1. EDA and data audit: Tc distribution by family, duplicate formulas, row alignment of the two files.
2. Baselines: Hamidieh XGBoost (published settings), tuned XGBoost, TabPFN-3.5 on engineered and on composition features.
3. Learning curves and uncertainty evaluation.
4. Discovery loop, including the hard variant.
5. "Why" analysis: where TabPFN wins/loses (by data size, Tc band, family), calibration.
6. Demo, README, figures, submission description, short video.
