# Submission: Hunting superconductors with TabPFN-3.5's predictive distribution

**One line:** We use TabPFN-3.5's per-material predictive distribution to run a simulated
superconductor search. It finds ~40× more top-1% materials than random search, and about
twice as many as greedy XGBoost or greedy TabPFN. Every number traces back to code and cached
API outputs that anyone can re-run without a key.

## What we did

The UCI superconductivity dataset (21,263 records from NIMS SuperCon) comes with a well-known
benchmark: Hamidieh (2018), XGBoost, RMSE ≈ 9.5 K, one global error bar. The paper suggests
the model could help researchers narrow the search for high-Tc materials, but it never tests
that. We do.

1. **Honest benchmark.** We first replicate the paper exactly: our XGBoost scores 9.42 K
   against the published 9.5 K. We then show the data is 21,263 rows but only 15,164
   materials: 35% of the paper's test rows have an exact duplicate in training. So the
   primary benchmark groups duplicates, and XGBoost gets a nested tuning budget of 2.8 h.
   TabPFN-3.5, with no tuning, beats tuned XGBoost on 25/25 grouped splits (8.97 vs 9.90 K),
   in every Tc band and family, and at every training size from 100 rows to the full set.
2. **Uncertainty per material.** TabPFN's 95% intervals cover 95.8% of unseen materials
   while being 38% narrower than conformal XGBoost. CRPS is 28% lower, and P(Tc > 77 K) has
   Brier 0.037 vs 0.050. The widths adapt: 9 K for low-Tc materials, 43 K for
   high-Tc cuprates.
3. **Simulated discovery.** We start from 50 measured materials, measure 10 more per round,
   and look for the top 1%. There are 152 targets among 15,147 materials, Tc ≥ 119 K. In 200
   experiments:
   - TabPFN with expected improvement finds **79.6 targets** on average.
   - Greedy TabPFN finds 42.5, greedy XGBoost 35.6, a Gaussian process with the same EI rule
     28.8, and random search 2.1.
   - On a hard non-cuprate pool (mostly iron-based targets) the scores are 53.9 vs 26.7, 33.6,
     20.2 and 2.3.

   EI and greedy TabPFN share the same model and first fit, so the gap is the value of the
   distribution itself. Greedy search stalls on very good but not-quite-target families. The
   model is right about them; the mean just can't see past them. EI scores the upper tail
   and never stalled in 20 runs.
4. **Why.** Hamidieh's top feature turns out to be largely a cuprate detector.
   TabPFN's interval width tracks how much Tc varies among similar materials (ρ = 0.83).
   That makes it a good signal inside the data, but not a novelty detector.
5. **Demo.** Type a formula and get Tc with 80%/95% intervals, P(Tc > 77 K) and the nearest
   training material. Known materials are held out, so the prediction is honest.

## How TabPFN-3.5 is used

- The Prior Labs API, `model_path="v3.5_default"` (checkpoint 20260909), regression.
- `output_type="main"` with a custom 107- or 146-level quantile grid gives the mean and the
  full predictive quantile function in one request. We verified this against
  `output_type="full"`: CRPS agrees to a median 0.0009 K.
- Expected improvement, P(top 1%) and P(Tc > 77 K) are all computed from that grid.
- TabPFN-3.5-Thinking was tested on 10 splits. It was worse here (9.69 vs 8.93 K), and we
  report it as such.
- 1,305 API requests in total, all through a budget guard. Every output is cached with an
  input fingerprint, so `make reproduce` rebuilds every table, figure and notebook with no
  API key, and CI checks it on a fresh Ubuntu clone.

## What we report against ourselves

- No model extrapolates to an unseen family. TabPFN's 95% intervals cover only 60% of
  held-out iron-based materials.
- Greedy XGBoost finds its first target sooner.
- EI's lead over greedy XGBoost on the hard pool holds on only 6/10 seeds.
- Discovery is simulated on known materials.

**Repo:** https://github.com/qeremeren/Superconductivity-Data-Analysis (Apache 2.0; data
CC BY 4.0, downloaded by script).
