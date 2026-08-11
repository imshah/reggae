"""Configuration: paths, model choices, provider selection, cost guard.

Config is persisted as JSON under the data directory and merged over defaults,
so edits survive across runs. Environment variables can override the data root
(DOCMIND_DATA) for testing.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# --- paths -----------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DOCMIND_DATA", PKG_DIR / "data"))
CONFIG_PATH = DATA_DIR / "config.json"

LANCE_DIR = DATA_DIR / "lancedb"
IMAGES_DIR = DATA_DIR / "images"          # extracted diagram images, per doc_id
MANIFEST_PATH = DATA_DIR / "manifest.json"
DIAGRAM_OUT_DIR = DATA_DIR / "diagrams"   # generated mermaid + renders
CHATS_DIR = DATA_DIR / "chats"            # persisted chat sessions (UI)


def ensure_dirs() -> None:
    for d in (DATA_DIR, LANCE_DIR, IMAGES_DIR, DIAGRAM_OUT_DIR, CHATS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --- config ----------------------------------------------------------------


@dataclass
class Config:
    # Ollama (local)
    ollama_host: str = "http://localhost:11434"
    embed_model: str = "qwen3-embedding"
    local_chat_model: str = "qwen3:14b"
    vision_model: str = "qwen2.5vl"

    # Remote provider selection
    remote_provider: str = "claude"       # "claude" | "kimi"

    # Claude
    claude_model: str = "claude-opus-4-8"          # default escalation model
    claude_heavy_model: str = "claude-opus-4-8"    # gaps/critique may pin stronger
    claude_effort: str = "high"

    # Kimi (Moonshot, OpenAI-compatible)
    kimi_base_url: str = "https://api.moonshot.ai/v1"
    kimi_model: str = "kimi-k3-turbo-preview"
    kimi_heavy_model: str = "kimi-k3-turbo-preview"

    # Retrieval
    top_k: int = 8
    chunk_tokens: int = 800
    chunk_overlap: int = 120
    active_group: str = "default"         # default group scope for queries/ingest

    # Cost guard (USD)
    budget_cap: float = 5.0               # per-session soft cap
    confirm_threshold: float = 0.10       # confirm before any call estimated above this

    @classmethod
    def load(cls) -> "Config":
        """Resolve settings with precedence: env var > config.json > defaults.

        Any field can be overridden by an environment variable named
        `DOCMIND_<FIELD>` (e.g. DOCMIND_CLAUDE_MODEL, DOCMIND_REMOTE_PROVIDER),
        which is convenient for keeping everything in one `.env`. API keys stay
        in their standard vars (ANTHROPIC_API_KEY / KIMI_API_KEY), not here.
        """
        cfg = cls()
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text())
                known = {f.name for f in fields(cls)}
                for k, v in data.items():
                    if k in known:
                        setattr(cfg, k, v)
            except (json.JSONDecodeError, OSError):
                pass
        # env overrides (from the real environment or a loaded .env)
        for f in fields(cls):
            env_key = "DOCMIND_" + f.name.upper()
            raw = os.environ.get(env_key)
            if raw:
                try:
                    setattr(cfg, f.name, _coerce(getattr(cfg, f.name), raw))
                except (ValueError, TypeError):
                    pass
        return cfg

    @classmethod
    def env_overrides(cls) -> set[str]:
        """Field names currently overridden by a DOCMIND_* env var."""
        return {
            f.name for f in fields(cls)
            if os.environ.get("DOCMIND_" + f.name.upper())
        }

    def save(self) -> None:
        ensure_dirs()
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))

    def set(self, key: str, value: str) -> None:
        """Set a key from a string value, coercing to the field's type."""
        known = {f.name for f in fields(self)}
        if key not in known:
            raise KeyError(key)
        setattr(self, key, _coerce(getattr(self, key), value))
        self.save()


def _coerce(current: object, value: str) -> object:
    """Coerce a string to the type of `current`."""
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value
