"""Reusable Streamlit render helpers."""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from docmind.engine import RemoteEstimate
from docmind.store.vectordb import Hit
from docmind.tasks.context import sources_list


def sources_expander(hits: list[Hit]) -> None:
    if not hits:
        return
    srcs = sources_list(hits)
    with st.expander(f"Sources ({len(srcs)})", expanded=False):
        for s in srcs:
            st.markdown(f"- `{s}`")


def cost_caption(estimate: RemoteEstimate, session_spent: float, cap: float) -> str:
    return (
        f"≈ ${estimate.est_usd:.4f} via {estimate.provider}/{estimate.model} · "
        f"session ${session_spent:.4f} / cap ${cap:.2f}"
    )


def render_svg(svg_path: str) -> bool:
    """Render a local SVG inline (no scripts, no iframe, no internet). Returns
    True if it rendered, False if the file was missing."""
    p = Path(svg_path)
    if not p.exists():
        return False
    b64 = base64.b64encode(p.read_text().encode()).decode()
    st.html(
        f'<img src="data:image/svg+xml;base64,{b64}" '
        f'style="max-width:100%;height:auto"/>'
    )
    return True


def render_mermaid(code: str, out: dict, key: str | None = None) -> None:
    """Render a generated diagram: the mmdc-produced SVG inline (local), the
    source, and download buttons.

    `key` must be unique per rendered diagram in a single run — multiple diagrams
    in the transcript would otherwise collide on auto-generated widget IDs. If
    omitted, a per-run sequence counter is used (reset at the top of main()).
    """
    if key is None:
        n = st.session_state.get("_mmd_seq", 0)
        st.session_state["_mmd_seq"] = n + 1
        key = f"mmd{n}"

    rendered = bool(out.get("svg")) and render_svg(out["svg"])
    if not rendered and out.get("render_error"):
        st.warning(f"Local render (mmdc) failed: {out['render_error']}")

    with st.expander("Mermaid source", expanded=not rendered):
        st.code(code, language="mermaid")

    cols = st.columns(2)
    cols[0].download_button("Download .mmd", code, file_name="diagram.mmd", key=f"{key}_mmd")
    if out.get("svg") and Path(out["svg"]).exists():
        cols[1].download_button(
            "Download .svg", Path(out["svg"]).read_text(), file_name="diagram.svg",
            key=f"{key}_svg",
        )
