.PHONY: setup data splits audit benchmark uncertainty discovery why why-xgb gp-discovery xgb-phase3 retrain-local demo readme notebooks reproduce test lint check-api requirements

setup:  ## Create the locked environment (Python 3.12 via uv)
	uv sync

data:  ## Download and verify the UCI superconductivity data into data/raw/
	uv run python -m src.data

splits:  ## Regenerate the saved train/test splits in data/splits/
	uv run python -m src.splits

audit:  ## Phase 1 data audit and figures (no API calls) -> results/01_audit/
	uv run python -m experiments.01_data_audit > /dev/null
	uv run python -m experiments.01_figures

benchmark:  ## Phase 2 tables from committed predictions; checks every cached TabPFN result
	uv run python -m experiments.02_tabpfn
	uv run python -m experiments.02_metrics > /dev/null
	uv run python -m experiments.02_figures

uncertainty:  ## Phase 3 tables and figures from committed files; checks every cached TabPFN result
	uv run python -m experiments.03_tabpfn_learning_curves
	uv run python -m experiments.03_evaluate --require-all
	uv run python -m experiments.03_figures

discovery:  ## Phase 4 tables and figures from committed run records (integrity-checked, no API)
	uv run python -m experiments.04_evaluate --check
	uv run python -m experiments.04_figures

xgb-phase3:  ## Slow, no API key: Phase 3 XGBoost learning curves and uncertainty baselines (resumable)
	uv run python -m experiments.03_xgb_learning_curves
	uv run python -m experiments.03_xgb_uncertainty

why:  ## Phase 5 "why" analysis from committed results (no API, no fitting)
	uv run python -m experiments.05_where_wins
	uv run python -m experiments.05_stalls
	uv run python -m experiments.05_intervals

why-xgb:  ## No API key, ~1-2 min: Phase 5 cuprate-detector test (refits XGBoost; outputs committed)
	uv run python -m experiments.05_cuprate_detector

gp-discovery:  ## No API key, ~1 min: rerun the Phase 5 GP-EI discovery baseline (records are committed)
	uv run python -m experiments.05_gp_discovery

demo:  ## Phase 6 demo showcase table and figure from the committed cache (no API)
	uv run python -m demo.showcase

readme:  ## Render the README tables from results/ (tests/test_readme.py checks they match)
	uv run python -m experiments.06_readme

retrain-local:  ## Optional, slow: refit local baselines (~20 min) and nested XGBoost tuning (~3 h)
	uv run python -m experiments.02_local_models
	uv run python -m experiments.02_xgb_tuning --trials 26

notebooks:  ## Execute every notebook in place (they only read results/)
	uv run --group notebooks python -c "import glob, nbclient, nbformat; [nbformat.write(nbclient.execute(nbformat.read(p, 4), cwd='notebooks', record_timing=False), p) for p in sorted(glob.glob('notebooks/*.ipynb'))]"

reproduce:  ## Rebuild everything from committed caches; no API key needed (grows each phase)
	$(MAKE) data
	$(MAKE) splits
	$(MAKE) audit
	$(MAKE) benchmark
	$(MAKE) uncertainty
	$(MAKE) discovery
	$(MAKE) why
	$(MAKE) demo
	$(MAKE) readme
	$(MAKE) notebooks
	uv run pytest

test:
	uv run pytest

lint:
	uv run ruff check .

check-api:  ## Free cost estimates; add LIVE=1 for the live check (~20k tokens)
	uv run python -m experiments.00_check_api $(if $(LIVE),--live,)

requirements:  ## Export pinned requirements.txt for pip users
	uv export --no-hashes --no-dev --no-emit-project -o requirements.txt
