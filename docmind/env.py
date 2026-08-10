"""Load environment variables from a .env file.

Loaded once on package import so API keys (ANTHROPIC_API_KEY, KIMI_API_KEY) can
live in a local `.env` instead of being exported each session. Real exported
environment variables always win — .env only fills in what's unset.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    candidates = [REPO_ROOT / ".env", Path.cwd() / ".env"]
    try:
        from dotenv import find_dotenv, load_dotenv

        found = find_dotenv(usecwd=True)  # walk up from CWD
        if found:
            load_dotenv(found, override=False)
        for p in candidates:
            if p.exists():
                load_dotenv(p, override=False)
    except ImportError:
        _manual_load(candidates)


def _manual_load(paths: list[Path]) -> None:
    """Minimal .env parser used if python-dotenv isn't installed yet."""
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, val)  # exported vars win
