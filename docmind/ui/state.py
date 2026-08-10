"""Session bootstrap: build the Engine and ChatStore once per Streamlit session.

Streamlit reruns the whole script on every interaction, so we cache the Engine
in st.session_state keyed by a signature of the mutable config fields. Changing
config in the sidebar changes the signature and rebuilds the Engine (new Ollama
client / LanceDB handle); otherwise the same instance is reused across reruns.
"""
from __future__ import annotations

from dataclasses import fields

import streamlit as st

from docmind.config import Config, ensure_dirs
from docmind.engine import Engine
from docmind.history import ChatStore

# config fields that require rebuilding the Engine when they change
_REBUILD_KEYS = ("ollama_host", "embed_model", "local_chat_model", "vision_model")


def _config_signature(cfg: Config) -> tuple:
    return tuple(getattr(cfg, k) for k in _REBUILD_KEYS)


def get_engine() -> Engine:
    ensure_dirs()
    cfg = Config.load()  # always reflect the on-disk config
    sig = _config_signature(cfg)
    if st.session_state.get("_engine_sig") != sig or "_engine" not in st.session_state:
        eng = Engine(cfg)
        # preserve spend across a rebuild within the same session
        eng.session_spent = st.session_state.get("_session_spent", 0.0)
        st.session_state["_engine"] = eng
        st.session_state["_engine_sig"] = sig
    else:
        # keep the live cfg in sync (non-rebuild fields like top_k, budget_cap)
        st.session_state["_engine"].cfg = cfg
    return st.session_state["_engine"]


def get_store() -> ChatStore:
    if "_chatstore" not in st.session_state:
        st.session_state["_chatstore"] = ChatStore()
    return st.session_state["_chatstore"]


def persist_spend(eng: Engine) -> None:
    st.session_state["_session_spent"] = eng.session_spent


def config_keys() -> list[str]:
    return [f.name for f in fields(Config)]
