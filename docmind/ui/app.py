"""docmind Streamlit UI — a thin browser front-end over docmind.engine.

Run via `docmind ui` (or `streamlit run docmind/ui/app.py`).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from docmind.config import DATA_DIR, Config, ensure_dirs
from docmind.engine import Aborted, BudgetExceeded, Engine
from docmind.history import Message
from docmind.ingest import parsers
from docmind.llm.provider import AVAILABLE_PROVIDERS
from docmind.llm.router import Route
from docmind.tasks import diagram as diagram_task
from docmind.tasks import gaps as gaps_task
from docmind.tasks import qa as qa_task
from docmind.tasks.context import sources_list
from docmind.ui import components as C
from docmind.ui.state import get_engine, get_store, persist_spend

UPLOAD_DIR = DATA_DIR / "uploads"

st.set_page_config(page_title="docmind", page_icon="📄", layout="wide")


# --- session bootstrap -----------------------------------------------------


def _ensure_active_session(store) -> str:
    if st.session_state.get("active_session_id"):
        # verify it still exists
        if store.load(st.session_state["active_session_id"]):
            return st.session_state["active_session_id"]
    sessions = store.list_sessions()
    sid = sessions[0].id if sessions else store.new_session().id
    st.session_state["active_session_id"] = sid
    return sid


# --- sidebar ---------------------------------------------------------------


ALL_GROUPS_LABEL = "All groups"


def _sidebar_group(eng: Engine) -> None:
    """Group scope selector — sets st.session_state['ui_group'] for the whole app."""
    st.subheader("🗂️ Group")
    options = [ALL_GROUPS_LABEL] + eng.list_groups()
    current = st.session_state.get("ui_group", eng.cfg.active_group)
    if current not in options:
        current = eng.cfg.active_group if eng.cfg.active_group in options else ALL_GROUPS_LABEL
    choice = st.selectbox(
        "Scope queries & uploads to", options, index=options.index(current),
        help="Answers and analysis only use documents in this group. "
        "'All groups' searches everything.",
    )
    st.session_state["ui_group"] = choice
    st.caption(
        "Isolated to this group — no cross-group context." if choice != ALL_GROUPS_LABEL
        else "Searching across every group."
    )


def _sidebar_documents(eng: Engine) -> None:
    st.subheader("📚 Documents")
    scope = _selected_group()  # None = all groups, else a name
    default_ingest_group = scope or "default"

    uploads = st.file_uploader(
        "Add documents",
        type=[e.lstrip(".") for e in sorted(parsers.SUPPORTED)],
        accept_multiple_files=True,
    )
    ingest_group = st.text_input(
        "Ingest into group", value=default_ingest_group,
        help="Tag for the uploaded files. Type a new name to create a group.",
    ).strip() or "default"
    if uploads and st.button("Ingest uploaded", use_container_width=True):
        ensure_dirs()
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with st.status(f"Ingesting into '{ingest_group}'…", expanded=True) as status:
            log = lambda m: status.write(m)
            for uf in uploads:
                dest = UPLOAD_DIR / uf.name
                dest.write_bytes(uf.getbuffer())
                try:
                    eng.ingest_path(dest, log, groups=[ingest_group])
                except Exception as e:  # noqa: BLE001 - surface any parse error
                    log(f"error on {uf.name}: {e}")
            status.update(label="Ingest complete", state="complete")
        st.rerun()

    docs = eng.manifest.all(group=scope)
    if not docs:
        st.caption("No documents in this scope yet.")
    all_groups = eng.list_groups()
    for d in docs:
        name = Path(d.source_path).name
        c1, c2 = st.columns([5, 1])
        c1.caption(f"**{name}** · `{', '.join(d.groups)}` · {d.chunk_count} chunks · {d.diagram_count} diagrams")
        if c2.button("🗑", key=f"rm_{d.doc_id}", help="Remove"):
            eng.remove(d.doc_id, log=lambda m: None)
            st.rerun()
        with c1.popover("Edit groups"):
            options = sorted(set(all_groups) | set(d.groups))
            selected = st.multiselect(
                "Member of", options, default=d.groups, key=f"grp_{d.doc_id}",
                help="A document can belong to several groups.",
            )
            new_group = st.text_input("Add a new group", key=f"newgrp_{d.doc_id}").strip()
            if st.button("Save groups", key=f"grpbtn_{d.doc_id}"):
                target = set(selected) | ({new_group} if new_group else set())
                for g in target - set(d.groups):
                    eng.add_doc_to_group(d.doc_id, g)
                for g in set(d.groups) - target:
                    eng.remove_doc_from_group(d.doc_id, g)
                st.rerun()

    if docs and st.button("Re-index all (this scope)", use_container_width=True):
        with st.status("Re-indexing…", expanded=True) as status:
            for d in docs:
                p = Path(d.source_path)
                if p.exists():
                    eng.ingest_path(p, log=lambda m: status.write(m), force=True, groups=d.groups)
                else:
                    status.write(f"missing source, skipped: {p}")
            status.update(label="Re-index complete", state="complete")
        st.rerun()


def _selected_group() -> str | None:
    """Resolve the sidebar selection into a retrieval scope: None = all groups."""
    label = st.session_state.get("ui_group", ALL_GROUPS_LABEL)
    return None if label == ALL_GROUPS_LABEL else label


def _sidebar_config(eng: Engine) -> None:
    st.subheader("⚙️ Configuration")
    cfg = Config.load()

    def maybe_set(key: str, value) -> None:
        if str(getattr(cfg, key)) != str(value):
            cfg.set(key, str(value))

    prov = st.selectbox(
        "Remote provider", list(AVAILABLE_PROVIDERS),
        index=list(AVAILABLE_PROVIDERS).index(cfg.remote_provider)
        if cfg.remote_provider in AVAILABLE_PROVIDERS else 0,
    )
    maybe_set("remote_provider", prov)

    maybe_set("local_chat_model", st.text_input("Local chat model", cfg.local_chat_model))
    maybe_set("top_k", st.number_input("top_k (chunks retrieved)", 1, 50, cfg.top_k))

    with st.expander("Remote models"):
        maybe_set("claude_model", st.text_input("Claude model (ask)", cfg.claude_model))
        maybe_set("claude_heavy_model", st.text_input("Claude model (gaps/critique)", cfg.claude_heavy_model))
        maybe_set("claude_effort", st.selectbox(
            "Claude effort", ["low", "medium", "high", "xhigh", "max"],
            index=["low", "medium", "high", "xhigh", "max"].index(cfg.claude_effort)
            if cfg.claude_effort in ["low", "medium", "high", "xhigh", "max"] else 2,
        ))
        maybe_set("kimi_model", st.text_input("Kimi model", cfg.kimi_model))

    with st.expander("Chunking (needs re-index)"):
        st.caption("Changing these requires **Re-index all** to affect existing docs.")
        maybe_set("chunk_tokens", st.number_input("chunk_tokens", 100, 4000, cfg.chunk_tokens, step=50))
        maybe_set("chunk_overlap", st.number_input("chunk_overlap", 0, 1000, cfg.chunk_overlap, step=10))

    with st.expander("Cost guard"):
        maybe_set("budget_cap", st.number_input("Session budget cap ($)", 0.0, 1000.0, float(cfg.budget_cap), step=0.5))
        maybe_set("confirm_threshold", st.number_input("Confirm above ($)", 0.0, 100.0, float(cfg.confirm_threshold), step=0.05))

    st.caption(f"Session spent: **${eng.session_spent:.4f}**")


def _sidebar_history(store) -> None:
    st.subheader("💬 Chats")
    if st.button("➕ New chat", use_container_width=True):
        st.session_state["active_session_id"] = store.new_session().id
        st.rerun()

    active = st.session_state.get("active_session_id")
    for s in store.list_sessions():
        is_active = s.id == active
        label = ("▶ " if is_active else "") + s.title
        if st.button(label, key=f"sess_{s.id}", use_container_width=True):
            st.session_state["active_session_id"] = s.id
            st.rerun()

    if active:
        with st.expander("Manage current chat"):
            sess = store.load(active)
            new_title = st.text_input("Rename", sess.title if sess else "")
            if st.button("Save title") and sess and new_title:
                store.rename(active, new_title)
                st.rerun()
            st.download_button(
                "Export .md",
                store.export_markdown(active),
                file_name=f"{(sess.title if sess else 'chat')}.md",
            )
            if st.button("🗑 Delete chat"):
                store.delete(active)
                st.session_state.pop("active_session_id", None)
                st.rerun()


# --- transcript rendering --------------------------------------------------


def _render_message(msg) -> None:
    with st.chat_message(msg.role):
        artifact = getattr(msg, "artifact", None)
        if msg.role == "assistant" and artifact and Path(artifact).exists():
            C.render_svg(artifact)
            with st.expander("Mermaid source"):
                st.code(diagram_task.extract_mermaid(msg.content), language="mermaid")
        elif msg.role == "assistant" and "```mermaid" in msg.content:
            st.code(diagram_task.extract_mermaid(msg.content), language="mermaid")
        else:
            st.markdown(msg.content)
        meta_bits = [b for b in (msg.route, msg.model) if b]
        if msg.cost:
            meta_bits.append(f"${msg.cost:.4f}")
        if meta_bits:
            st.caption(" · ".join(meta_bits))
        if msg.sources:
            with st.expander(f"Sources ({len(msg.sources)})"):
                for s in msg.sources:
                    st.markdown(f"- `{s}`")


def _render_transcript(store, sid: str) -> None:
    sess = store.load(sid)
    if not sess:
        return
    for msg in sess.messages:
        _render_message(msg)


# --- actions ---------------------------------------------------------------


def _run_remote_or_notice(eng: Engine, system: str, user: str, *, heavy: bool,
                          provider: str | None, output_guess: int, max_tokens: int):
    """Returns (text, provider, model, cost) or (None, ...) if not runnable."""
    est = eng.estimate_remote(system, user, heavy=heavy, provider_name=provider,
                              output_guess=output_guess)
    if not est.available:
        st.info(f"{est.reason}")
        return None, est.provider, est.model, None
    if est.over_cap:
        st.error(
            f"Estimated ${est.est_usd:.4f} would exceed the session budget cap. "
            "Raise it in the sidebar (Cost guard) to proceed."
        )
        return None, est.provider, est.model, None
    st.caption(C.cost_caption(est, eng.session_spent, eng.cfg.budget_cap))
    try:
        with st.spinner(f"Calling {est.provider}/{est.model}…"):
            comp = eng.run_remote(
                system, user, heavy=heavy, log=lambda m: None,
                confirm=lambda _m: True, provider_name=provider,
                output_guess=output_guess, max_tokens=max_tokens,
            )
    except (Aborted, BudgetExceeded) as e:
        st.error(str(e))
        return None, est.provider, est.model, None
    persist_spend(eng)
    from docmind.llm import pricing
    cost = pricing.estimate_cost(
        comp.provider, comp.model, comp.usage.input_tokens, comp.usage.output_tokens,
        comp.usage.cache_read_tokens, comp.usage.cache_write_tokens,
    )
    return comp.text, comp.provider, comp.model, cost


def _generate_ui(eng: Engine, system: str, user: str, *, heavy: bool, provider: str,
                 output_guess: int, max_tokens: int, override: str):
    """Generate for analysis/diagram: remote if possible/allowed, else local.

    Honours the Route toggle (Local forces local) and the cost guard (over-cap or
    unavailable → local fallback with a notice). Always returns text.
    Returns (text, route, provider, model, cost).
    """
    force_local = override == "Local"
    if not force_local:
        est = eng.estimate_remote(system, user, heavy=heavy, provider_name=provider,
                                  output_guess=output_guess)
        if est.available and not est.over_cap:
            st.caption(C.cost_caption(est, eng.session_spent, eng.cfg.budget_cap))
            try:
                with st.spinner(f"Calling {est.provider}/{est.model}…"):
                    res = eng.generate(system, user, heavy=heavy, log=lambda m: None,
                                       confirm=lambda _m: True, provider_name=provider,
                                       output_guess=output_guess, max_tokens=max_tokens)
                persist_spend(eng)
                return res.text, res.route, res.provider, res.model, res.cost
            except Exception as e:  # noqa: BLE001
                st.warning(f"Remote call failed ({e}); using local model.")
        else:
            reason = est.reason or "would exceed the session budget cap"
            st.info(f"{reason} — using the local model.")

    with st.spinner(f"Generating locally ({eng.cfg.local_chat_model})…"):
        res = eng.generate(system, user, heavy=heavy, log=lambda m: None,
                           confirm=lambda _m: True, force_local=True)
    return res.text, res.route, res.provider, res.model, res.cost


def _answer_ask(eng: Engine, store, sid: str, prompt: str, override: str, provider: str) -> None:
    """Produce the assistant turn for a question (user turn handled by dispatcher)."""
    force = {"Local": Route.LOCAL, "Remote": Route.REMOTE}.get(override)
    route = eng.route(prompt, force)
    hits = eng.retrieve(prompt, group=_selected_group())
    system, user = qa_task.build_prompt(prompt, hits)
    srcs = sources_list(hits)

    with st.chat_message("assistant"):
        if route == Route.LOCAL:
            text = st.write_stream(eng.answer_local_stream(system, user))
            st.caption(f"local · {eng.cfg.local_chat_model}")
            store.append(sid, Message(role="assistant", content=text, route="local",
                                      model=eng.cfg.local_chat_model, cost=0.0, sources=srcs))
        else:
            text, prov, model, cost = _run_remote_or_notice(
                eng, system, user, heavy=False, provider=provider,
                output_guess=800, max_tokens=4096,
            )
            if text is None:  # fall back to local
                st.caption("falling back to local")
                text = st.write_stream(eng.answer_local_stream(system, user))
                store.append(sid, Message(role="assistant", content=text, route="local",
                                          model=eng.cfg.local_chat_model, cost=0.0, sources=srcs))
            else:
                st.markdown(text)
                store.append(sid, Message(role="assistant", content=text, route="remote",
                                          provider=prov, model=model, cost=cost, sources=srcs))
        if srcs:
            with st.expander(f"Sources ({len(srcs)})"):
                for s in srcs:
                    st.markdown(f"- `{s}`")


def _answer_analysis(eng: Engine, store, sid: str, kind: str, scope: str, provider: str,
                     override: str) -> None:
    """Produce the assistant turn for gaps/critique (user turn handled by dispatcher)."""
    hits = eng.retrieve(scope or "architecture process ownership risk", k=eng.cfg.top_k * 2, group=_selected_group())
    if not hits:
        with st.chat_message("assistant"):
            st.warning("No indexed content to analyse. Add documents first.")
        return
    if kind == "critique":
        system, user = gaps_task.build_critique_prompt(hits, scope or None)
    else:
        system, user = gaps_task.build_gaps_prompt(hits, scope or None)
    srcs = sources_list(hits)
    with st.chat_message("assistant"):
        text, route, prov, model, cost = _generate_ui(
            eng, system, user, heavy=True, provider=provider,
            output_guess=2500, max_tokens=6000, override=override,
        )
        st.markdown(text or "_(no output)_")
        st.caption(f"{route} · {model} · ${cost:.4f}")
        store.append(sid, Message(role="assistant", content=text, route=f"{kind}/{route}",
                                  provider=prov, model=model, cost=cost, sources=srcs))


def _answer_diagram(eng: Engine, store, sid: str, desc: str, provider: str,
                    mindmap: bool, override: str, freeform: bool) -> None:
    """Produce the assistant turn for a diagram/mind map (user turn handled by dispatcher)."""
    kind = "mindmap" if mindmap else "diagram"
    if freeform:
        system, user = diagram_task.build_freeform_prompt(desc, mindmap)
    else:
        hits = eng.retrieve(desc, k=eng.cfg.top_k * 2, group=_selected_group())
        system, user = diagram_task.build_diagram_prompt(desc, hits, mindmap)
    with st.chat_message("assistant"):
        text, route, prov, model, cost = _generate_ui(
            eng, system, user, heavy=False, provider=provider,
            output_guess=800, max_tokens=4096, override=override,
        )
        code = diagram_task.extract_mermaid(text)
        out = diagram_task.write_outputs(desc, code, render=True)
        C.render_mermaid(code, out)
        st.caption(f"{route} · {model} · ${cost:.4f}")
        store.append(sid, Message(role="assistant", content=f"```mermaid\n{code}\n```",
                                  route=f"{kind}/{route}", provider=prov, model=model,
                                  cost=cost, artifact=out.get("svg")))


# --- slash-command dispatch ------------------------------------------------

_COMMAND_HELP = (
    "**Commands** — type in the chat box:\n"
    "- `/gaps [scope]` — gap analysis for a senior tech-management lens\n"
    "- `/critique [scope]` — architecture / design review\n"
    "- `/diagram <description>` — Mermaid diagram grounded in your docs\n"
    "- `/mindmap <topic>` — Mermaid mind map grounded in your docs\n"
    "- add `--freeform` to `/diagram` or `/mindmap` to build from your text, "
    "ignoring the corpus\n"
    "- `/help` — show this list\n\n"
    "Anything without a leading `/` is answered as a question. The **Route** and "
    "**Provider** controls above decide local vs. remote."
)


def _strip_flag(text: str, *flags: str) -> tuple[str, bool]:
    """Remove any of `flags` (as whole tokens) from text; return (clean, found)."""
    tokens = text.split()
    kept = [t for t in tokens if t.lower() not in flags]
    return " ".join(kept), len(kept) != len(tokens)


def _dispatch(eng: Engine, store, sid: str, prompt: str, override: str,
              provider: str) -> None:
    """Render + persist the user's turn, then route to the right producer."""
    with st.chat_message("user"):
        st.markdown(prompt)
    store.append(sid, Message(role="user", content=prompt))

    text = prompt.strip()
    if not text.startswith("/"):
        _answer_ask(eng, store, sid, text, override, provider)
        return

    cmd, _, rest = text.partition(" ")
    cmd = cmd[1:].lower()  # drop leading '/'
    rest = rest.strip()

    if cmd == "help":
        with st.chat_message("assistant"):
            st.info(_COMMAND_HELP)
    elif cmd == "ask":
        _answer_ask(eng, store, sid, rest, override, provider)
    elif cmd in ("gaps", "critique"):
        _answer_analysis(eng, store, sid, cmd, rest, provider, override)
    elif cmd in ("diagram", "mindmap"):
        desc, freeform = _strip_flag(rest, "--freeform", "-f")
        desc = desc.strip()
        if not desc:
            with st.chat_message("assistant"):
                st.warning(f"Give a description, e.g. `/{cmd} order fulfilment flow`.")
            return
        _answer_diagram(eng, store, sid, desc, provider,
                        mindmap=(cmd == "mindmap"), override=override, freeform=freeform)
    else:
        with st.chat_message("assistant"):
            st.warning(f"Unknown command `/{cmd}` — try `/help`.")


# --- main ------------------------------------------------------------------


def main() -> None:
    st.session_state["_mmd_seq"] = 0  # reset per-run keys for Mermaid download buttons
    eng = get_engine()
    store = get_store()
    sid = _ensure_active_session(store)

    with st.sidebar:
        _sidebar_group(eng)
        st.divider()
        _sidebar_documents(eng)
        st.divider()
        _sidebar_history(store)
        st.divider()
        _sidebar_config(eng)

    st.title("📄 docmind")

    # controls
    c1, c2 = st.columns(2)
    override = c1.radio("Route", ["Auto", "Local", "Remote"], horizontal=True)
    provider = c2.selectbox("Provider", list(AVAILABLE_PROVIDERS),
                            index=list(AVAILABLE_PROVIDERS).index(eng.cfg.remote_provider)
                            if eng.cfg.remote_provider in AVAILABLE_PROVIDERS else 0)

    st.caption(
        "Ask a question, or use a command: `/gaps` · `/critique` · `/diagram` · "
        "`/mindmap` · `/help`  (add `--freeform` to build diagrams from your text)."
    )

    _render_transcript(store, sid)

    prompt = st.chat_input("Ask a question, or type /help …")
    if prompt:
        _dispatch(eng, store, sid, prompt, override, provider)


main()
