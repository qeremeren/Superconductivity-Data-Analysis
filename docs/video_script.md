# Video script (about 2.5 minutes)

Screen recording with voice-over. Every visual is a file already in the repo; every number
is in the README.

| # | Time | On screen | Say |
|---|---|---|---|
| 1 | 0:00–0:15 | README title, then `results/01_audit/figures/tc_by_family.png` | "Superconductors that work above 77 K, the temperature of liquid nitrogen, are rare and valuable. The standard dataset has 21,000 measured superconductors, and the standard model, XGBoost from Hamidieh 2018, predicts Tc to about 9.5 K. That paper suggests the model could narrow the search for new high-Tc materials, but never tests it. We did, with TabPFN-3.5." |
| 2 | 0:15–0:40 | `results/01_audit/figures/duplicate_spread.png`, then the benchmark table in the README | "First, an honest benchmark. The 21,000 rows are only 15,000 materials, so 35% of the paper's test rows are duplicates of training rows. We group them so no material crosses train and test, and give XGBoost almost three hours of tuning. TabPFN, untuned, still wins on all 25 splits: 8.97 versus 9.90 kelvin." |
| 3 | 0:40–1:05 | `results/03_uncertainty/figures/calibration_curve.png` | "The paper gives one error bar for every material. TabPFN gives a full distribution per material. Its 95% intervals cover 95.8% of unseen materials, and they are 38% narrower than conformal XGBoost: tight for simple metals, wide for cuprates." |
| 4 | 1:05–1:40 | `results/04_discovery/figures/discovery_curves.png` | "Now the real test: a simulated discovery campaign. Start with 50 measured materials, measure 10 more per round, and look for the top 1%, Tc above 119 K. Random search finds 2 in 200 experiments. Greedy XGBoost finds 36. TabPFN with expected improvement, which uses the whole predictive distribution, finds 80. Same model, greedy on the mean: 42. So the distribution itself is doing the work." |
| 5 | 1:40–2:00 | `results/05_why/figures/stalls.png` | "Why does greedy stall? It keeps measuring YBCO-type materials around 90 kelvin. The model is right that they're good, but they're not good enough. Expected improvement looks at the upper tail and moves to the mercury and thallium cuprates, where the targets are." |
| 6 | 2:00–2:20 | Terminal: `uv run python -m demo.predict HgBa2Ca2Cu3O8 LaH10`, then `results/06_demo/figures/showcase.png` | "The demo: type a formula, get Tc with an interval, the chance it beats 77 K, and the nearest known material. Known materials are held out, so this is a real prediction. For LaH10, a high-pressure hydride the data can't describe, the model says it doesn't know." |
| 7 | 2:20–2:35 | README "Where it falls short", then a terminal running `make reproduce` | "We report the limits too: no model extrapolates to a family it has never seen, and greedy XGBoost finds its first hit sooner. Everything here rebuilds from one command, with no API key." |

Recording tips: run `uv run python -m demo.predict HgBa2Ca2Cu3O8 LaH10` once before
recording (it reads the cache, so it is instant and costs nothing). Open the PNGs from
`results/` at full size.
