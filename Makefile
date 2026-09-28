.PHONY: setup data splits audit notebooks test lint check-api requirements

setup:  ## Create the locked environment (Python 3.12 via uv)
	uv sync

data:  ## Download and verify the UCI superconductivity data into data/raw/
	uv run python -m src.data

splits:  ## Regenerate the saved train/test splits in data/splits/
	uv run python -m src.splits

audit:  ## Phase 1 data audit and figures (no API calls) -> results/01_audit/
	uv run python -m experiments.01_data_audit > /dev/null
	uv run python -m experiments.01_figures

notebooks:  ## Execute every notebook in place (they only read results/)
	uv run --group notebooks python -c "import glob, nbclient, nbformat; [nbformat.write(nbclient.execute(nbformat.read(p, 4), cwd='notebooks'), p) for p in sorted(glob.glob('notebooks/*.ipynb'))]"

test:
	uv run pytest

lint:
	uv run ruff check .

check-api:  ## Free cost estimates; add LIVE=1 for the live check (~20k tokens)
	uv run python -m experiments.00_check_api $(if $(LIVE),--live,)

requirements:  ## Export pinned requirements.txt for pip users
	uv export --no-hashes --no-dev --no-emit-project -o requirements.txt
