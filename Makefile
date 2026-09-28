.PHONY: setup data test lint check-api requirements

setup:  ## Create the locked environment (Python 3.12 via uv)
	uv sync

data:  ## Download and verify the UCI superconductivity data into data/raw/
	uv run python -m src.data

test:
	uv run pytest

lint:
	uv run ruff check .

check-api:  ## Free cost estimates; add LIVE=1 for one tiny live fit/predict (~10k tokens)
	uv run python experiments/00_check_api.py $(if $(LIVE),--live,)

requirements:  ## Export pinned requirements.txt for pip users
	uv export --no-hashes --no-dev --no-emit-project -o requirements.txt
