"""Diagram / mind map generation as Mermaid, grounded in retrieved context.

The model returns a single Mermaid code block; we extract it, write a .mmd and
a fenced markdown file, and optionally render to SVG via mermaid-cli (npx),
which uses the Node provisioned by mise.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from docmind.config import DIAGRAM_OUT_DIR, ensure_dirs
from docmind.store.vectordb import Hit
from docmind.tasks.context import format_context

SYSTEM_DIAGRAM = (
    "You produce Mermaid diagrams from technical documentation. Use ONLY facts "
    "present in the provided context excerpts; do not invent components or steps. "
    "Choose the most fitting Mermaid diagram type (flowchart, sequenceDiagram, "
    "erDiagram, etc.). Respond with EXACTLY ONE ```mermaid code block and nothing "
    "else — no prose before or after."
)

SYSTEM_MINDMAP = (
    "You produce Mermaid mindmap diagrams that organise a topic from technical "
    "documentation. Use ONLY facts present in the provided context excerpts. "
    "Respond with EXACTLY ONE ```mermaid code block using `mindmap` syntax and "
    "nothing else."
)

SYSTEM_DIAGRAM_FREEFORM = (
    "You produce Mermaid diagrams from the user's description. Choose the most "
    "fitting Mermaid diagram type (flowchart, sequenceDiagram, erDiagram, etc.). "
    "Respond with EXACTLY ONE ```mermaid code block and nothing else — no prose "
    "before or after."
)

SYSTEM_MINDMAP_FREEFORM = (
    "You produce Mermaid mindmap diagrams that organise the user's topic. "
    "Respond with EXACTLY ONE ```mermaid code block using `mindmap` syntax and "
    "nothing else."
)

_FENCE = re.compile(r"```(?:mermaid)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def build_diagram_prompt(description: str, hits: list[Hit], mindmap: bool) -> tuple[str, str]:
    context = format_context(hits)
    system = SYSTEM_MINDMAP if mindmap else SYSTEM_DIAGRAM
    user = f"Context excerpts:\n\n{context}\n\n---\nRequest: {description}"
    return system, user


def build_freeform_prompt(description: str, mindmap: bool) -> tuple[str, str]:
    """Diagram from the user's description directly, ignoring the corpus."""
    system = SYSTEM_MINDMAP_FREEFORM if mindmap else SYSTEM_DIAGRAM_FREEFORM
    return system, description


def extract_mermaid(text: str) -> str:
    m = _FENCE.search(text)
    code = m.group(1).strip() if m else text.strip()
    return code


def write_outputs(name_hint: str, mermaid: str, render: bool) -> dict[str, str]:
    ensure_dirs()
    slug = re.sub(r"[^a-z0-9]+", "-", name_hint.lower()).strip("-")[:40] or "diagram"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = DIAGRAM_OUT_DIR / f"{slug}-{stamp}"
    mmd = base.with_suffix(".mmd")
    md = base.with_suffix(".md")
    mmd.write_text(mermaid + "\n")
    md.write_text(f"```mermaid\n{mermaid}\n```\n")

    out = {"mmd": str(mmd), "md": str(md)}
    if render:
        svg = base.with_suffix(".svg")
        try:
            subprocess.run(
                ["npx", "-y", "@mermaid-js/mermaid-cli", "-i", str(mmd), "-o", str(svg)],
                check=True,
                capture_output=True,
                text=True,
            )
            out["svg"] = str(svg)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            detail = getattr(e, "stderr", "") or str(e)
            out["render_error"] = detail.strip()[:500]
    return out
