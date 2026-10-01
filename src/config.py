"""Environment and TabPFN API settings shared by all experiments.

The API key lives in TABPFN_TOKEN, set in the shell or in a git-ignored .env at
the repo root (see .env.example). Cached runs never need it. Live API calls are
opt-in via TABPFN_LIVE=1 or an experiment's --live flag.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
RESULTS_DIR = ROOT / "results"

# Pinned instead of "auto" so a server-side default change before the deadline
# cannot silently alter results. Through the API this selects TabPFN-3.5-Plus.
TABPFN_MODEL_PATH = "v3.5_default"
TABPFN_MODEL_VERSION = "v3.5"

TOKEN_URL = "https://platform.priorlabs.ai/account/api-keys"


def load_env(env_file: Path | None = None) -> None:
    """Load .env into os.environ without overriding variables already exported."""
    load_dotenv(env_file or ENV_FILE, override=False)


def live_requested(cli_flag: bool = False) -> bool:
    """Whether this run may call the TabPFN API instead of reading the cache."""
    load_env()
    return cli_flag or os.getenv("TABPFN_LIVE", "").strip().lower() in ("1", "true")


def record_spend() -> bool:
    """Whether billed requests are also appended to the committed spend record
    (set TABPFN_RECORD_SPEND=1; only the project author should)."""
    load_env()
    return os.getenv("TABPFN_RECORD_SPEND", "").strip().lower() in ("1", "true")


def require_tabpfn_token() -> None:
    """Fail early, with instructions, if an API call is about to run without a key.

    tabpfn-client reads TABPFN_TOKEN from the environment itself, so this only
    has to make sure .env has been loaded. The token is never returned or printed.
    """
    load_env()
    if not os.getenv("TABPFN_TOKEN", "").strip():
        raise RuntimeError(
            "TabPFN API calls need TABPFN_TOKEN. Copy .env.example to .env "
            f"and paste a key from {TOKEN_URL}, or export it in your shell."
        )
