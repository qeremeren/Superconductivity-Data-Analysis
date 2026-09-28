# Superconductor Tc discovery with TabPFN-3.5

Work in progress for the Prior Labs TabPFN-3.5 Hackathon. We predict the critical temperature
(Tc) of superconductors from chemical composition with TabPFN-3.5, benchmark it against XGBoost
(including the published Hamidieh 2018 baseline), and test whether its predictive distribution
can find materials above 77 K with fewer simulated experiments.

No results are reported yet. Every number in the final README will come from code in this repo
and a file in `results/`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). On macOS, XGBoost also needs `brew install libomp`.

```bash
make setup   # Python 3.12 environment from uv.lock
make data    # download and checksum-verify the UCI data into data/raw/
make test
```

pip users can install from `requirements.txt` (exported from the lock) into a Python 3.12 environment.

### TabPFN API key

Cached TabPFN outputs will be committed under `results/`, so reproducing the figures does not need
a key. To make live API calls:

```bash
cp .env.example .env   # then paste your key from https://platform.priorlabs.ai/account/api-keys
make check-api         # free cost estimates; `make check-api LIVE=1` adds one tiny live call
```

`.env` is git-ignored. Live calls are opt-in via `TABPFN_LIVE=1` or an experiment's `--live` flag.

## Data

"Superconductivty Data" from the UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/464/superconductivty+data), created by Kam Hamidieh from the
NIMS SuperCon database, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The data is downloaded by
`make data` and not redistributed in this repository.

## References

- K. Hamidieh, "A data-driven statistical model for predicting the critical temperature of a
  superconductor," *Computational Materials Science* 154 (2018) 346–354,
  doi:[10.1016/j.commatsci.2018.07.052](https://doi.org/10.1016/j.commatsci.2018.07.052).
  Code: https://github.com/khamidieh/predict_tc
- B. Jäger et al., "TabPFN-3.5: Technical Report," arXiv:2609.17895 (2026).
- N. Hollmann et al., "Accurate predictions on small data with a tabular foundation model,"
  *Nature* 637 (2025) 319–326, doi:10.1038/s41586-024-08328-6.

## License

Code: Apache License 2.0 (see `LICENSE`). Data: CC BY 4.0, see above.
