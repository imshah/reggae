"""docmind CLI + interactive REPL."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from docmind.config import Config, ensure_dirs
from docmind.engine import Aborted, BudgetExceeded, Engine
from docmind.llm.provider import AVAILABLE_PROVIDERS
from docmind.llm.router import Route
from docmind.tasks import diagram as diagram_task
from docmind.tasks import gaps as gaps_task
from docmind.tasks import qa as qa_task

app = typer.Typer(
    add_completion=False,
    help="Local-first document intelligence agent (add/ask/gaps/diagram) with "
    "swappable remote LLM providers.",
    no_args_is_help=True,
)
console = Console()


# --- shared helpers --------------------------------------------------------


def _engine() -> Engine:
    ensure_dirs()
    return Engine(Config.load())


def _log(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")


def _confirm(msg: str) -> bool:
    return Confirm.ask(f"[yellow]{msg}[/yellow]", default=False)


def _sources_panel(hits) -> None:
    from docmind.tasks.context import sources_list

    if hits:
        srcs = sources_list(hits)
        console.print(Panel("\n".join(f"• {s}" for s in srcs), title="Sources", expand=False))


# --- commands --------------------------------------------------------------


@app.command()
def add(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    force: bool = typer.Option(False, "--force", help="Re-ingest even if unchanged"),
) -> None:
    """Ingest documents (pdf/docx/txt/md), describing diagrams locally."""
    eng = _engine()
    console.print(f"[bold]Ingesting[/bold] {path}")
    try:
        results = eng.ingest_path(Path(path), _log, force=force)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    n_new = sum(1 for r in results if not r.skipped)
    console.print(
        f"[green]Done.[/green] {n_new} indexed, {len(results) - n_new} unchanged."
    )


@app.command(name="list")
def list_docs() -> None:
    """List indexed documents."""
    eng = _engine()
    docs = eng.manifest.all()
    if not docs:
        console.print("[dim]No documents indexed yet. Use `docmind add <path>`.[/dim]")
        return
    table = Table(title="Indexed documents")
    table.add_column("doc_id", style="cyan")
    table.add_column("source")
    table.add_column("chunks", justify="right")
    table.add_column("diagrams", justify="right")
    for d in docs:
        table.add_row(d.doc_id, d.source_path, str(d.chunk_count), str(d.diagram_count))
    console.print(table)
    console.print(f"[dim]Total chunks in store: {eng.store.count()}[/dim]")


@app.command()
def remove(doc_id: str = typer.Argument(..., help="doc_id from `list`")) -> None:
    """Remove a document and all its chunks/artifacts."""
    eng = _engine()
    rec = eng.remove(doc_id, _log)
    if rec is None:
        console.print(f"[red]No such doc_id: {doc_id}[/red]")
        raise typer.Exit(1)


@app.command()
def ask(
    question: List[str] = typer.Argument(..., help="Your question (quotes optional)"),
    provider: Optional[str] = typer.Option(None, "--provider", help="claude|kimi (for escalated calls)"),
    remote: bool = typer.Option(False, "--remote", help="Force remote provider"),
    local: bool = typer.Option(False, "--local", help="Force local model"),
) -> None:
    """Answer a question across the corpus, with citations."""
    eng = _engine()
    _run_ask(eng, " ".join(question), provider, remote, local)


@app.command()
def gaps(
    scope: Optional[str] = typer.Option(None, "--scope", help="Focus area"),
    provider: Optional[str] = typer.Option(None, "--provider", help="claude|kimi"),
    local: bool = typer.Option(False, "--local", help="Force the local model (no API key needed)"),
) -> None:
    """Gap analysis for a senior tech-management lens (remote, falls back to local)."""
    eng = _engine()
    _run_gaps(eng, scope, provider, critique=False, local=local)


@app.command()
def critique(
    scope: Optional[str] = typer.Option(None, "--scope", help="Focus area"),
    provider: Optional[str] = typer.Option(None, "--provider", help="claude|kimi"),
    local: bool = typer.Option(False, "--local", help="Force the local model (no API key needed)"),
) -> None:
    """Architecture critique / design review (remote, falls back to local)."""
    eng = _engine()
    _run_gaps(eng, scope, provider, critique=True, local=local)


@app.command()
def diagram(
    description: List[str] = typer.Argument(..., help="What to diagram (quotes optional)"),
    render: bool = typer.Option(False, "--render", help="Render to SVG via mermaid-cli"),
    provider: Optional[str] = typer.Option(None, "--provider", help="claude|kimi"),
    local: bool = typer.Option(False, "--local", help="Force the local model (no API key needed)"),
    freeform: bool = typer.Option(False, "--freeform", help="Build from your description, ignore the corpus"),
) -> None:
    """Generate a Mermaid diagram (grounded in the corpus; remote → local fallback)."""
    eng = _engine()
    _run_diagram(eng, " ".join(description), render, provider, mindmap=False, local=local, freeform=freeform)


@app.command()
def mindmap(
    topic: List[str] = typer.Argument(..., help="Topic to map (quotes optional)"),
    render: bool = typer.Option(False, "--render", help="Render to SVG via mermaid-cli"),
    provider: Optional[str] = typer.Option(None, "--provider", help="claude|kimi"),
    local: bool = typer.Option(False, "--local", help="Force the local model (no API key needed)"),
    freeform: bool = typer.Option(False, "--freeform", help="Build from your topic, ignore the corpus"),
) -> None:
    """Generate a Mermaid mind map (grounded in the corpus; remote → local fallback)."""
    eng = _engine()
    _run_diagram(eng, " ".join(topic), render, provider, mindmap=True, local=local, freeform=freeform)


@app.command()
def config(
    action: str = typer.Argument("show", help="show | get <key> | set <key> <value>"),
    key: Optional[str] = typer.Argument(None),
    value: Optional[str] = typer.Argument(None),
) -> None:
    """View or change configuration."""
    cfg = Config.load()
    if action == "show":
        _show_config(cfg)
    elif action == "get" and key:
        console.print(getattr(cfg, key, f"[red]unknown key: {key}[/red]"))
    elif action == "set" and key and value is not None:
        try:
            cfg.set(key, value)
            console.print(f"[green]{key} = {getattr(cfg, key)}[/green]")
        except KeyError:
            console.print(f"[red]unknown key: {key}[/red]")
    else:
        console.print("Usage: config show | config get <key> | config set <key> <value>")


@app.command()
def repl() -> None:
    """Interactive session."""
    _repl()


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", help="Port for the Streamlit server"),
) -> None:
    """Launch the Streamlit web UI."""
    import subprocess
    import sys
    from importlib import util as importlib_util

    if importlib_util.find_spec("streamlit") is None:
        console.print(
            "[red]Streamlit is not installed.[/red] Install the UI extra:\n"
            "  [cyan]pip install -e \".[ui]\"[/cyan]   (or re-run ./setup.sh)"
        )
        raise typer.Exit(1)

    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    console.print(f"[green]Starting docmind UI[/green] → http://localhost:{port}")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path),
             "--server.port", str(port)],
            check=True,
        )
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Streamlit exited with error: {e}[/red]")
        raise typer.Exit(1)


# --- task runners (shared by CLI + REPL) -----------------------------------


def _run_ask(eng: Engine, question: str, provider, remote: bool, local: bool) -> None:
    force = Route.REMOTE if remote else Route.LOCAL if local else None
    route = eng.route(question, force)
    hits = eng.retrieve(question)
    system, user = qa_task.build_prompt(question, hits)

    if route == Route.LOCAL:
        _log("routing: local (qwen)")
        answer = eng.answer_local(system, user)
    else:
        try:
            comp = eng.run_remote(
                system, user, heavy=False, log=_log, confirm=_confirm,
                provider_name=provider, output_guess=800,
            )
            answer = comp.text
        except (Aborted, BudgetExceeded) as e:
            console.print(f"[yellow]Skipped remote call: {e}. Falling back to local.[/yellow]")
            answer = eng.answer_local(system, user)

    console.print(Markdown(answer or "_(no answer)_"))
    _sources_panel(hits)


def _run_gaps(eng: Engine, scope, provider, critique: bool, local: bool = False) -> None:
    hits = eng.retrieve(scope or "architecture process ownership risk", k=eng.cfg.top_k * 2)
    if not hits:
        console.print("[yellow]No indexed content to analyse. Add documents first.[/yellow]")
        return
    if critique:
        system, user = gaps_task.build_critique_prompt(hits, scope)
    else:
        system, user = gaps_task.build_gaps_prompt(hits, scope)
    res = eng.generate(
        system, user, heavy=True, log=_log, confirm=_confirm, force_local=local,
        provider_name=provider, output_guess=2500, max_tokens=6000,
    )
    console.print(Markdown(res.text or "_(no output)_"))
    console.print(f"[dim]{res.route} · {res.model} · ${res.cost:.4f}[/dim]")
    _sources_panel(hits)


def _run_diagram(eng: Engine, description: str, render: bool, provider, mindmap: bool,
                 local: bool = False, freeform: bool = False) -> None:
    if freeform:
        hits = []
        system, user = diagram_task.build_freeform_prompt(description, mindmap)
    else:
        hits = eng.retrieve(description, k=eng.cfg.top_k * 2)
        system, user = diagram_task.build_diagram_prompt(description, hits, mindmap)
    res = eng.generate(
        system, user, heavy=False, log=_log, confirm=_confirm, force_local=local,
        provider_name=provider, output_guess=800,
    )
    mermaid = diagram_task.extract_mermaid(res.text)
    console.print(Panel(mermaid, title="Mermaid", expand=False))
    console.print(f"[dim]{res.route} · {res.model} · ${res.cost:.4f}[/dim]")
    out = diagram_task.write_outputs(description, mermaid, render)
    for k, v in out.items():
        style = "red" if k == "render_error" else "green"
        console.print(f"[{style}]{k}:[/{style}] {v}")


def _show_config(cfg: Config) -> None:
    from dataclasses import fields

    overrides = Config.env_overrides()
    table = Table(title="Configuration")
    table.add_column("key", style="cyan")
    table.add_column("value")
    table.add_column("source")
    for f in fields(cfg):
        src = "env (DOCMIND_*)" if f.name in overrides else "config.json/default"
        table.add_row(f.name, str(getattr(cfg, f.name)), src)
    console.print(table)
    if overrides:
        console.print(
            "[dim]Keys marked 'env' are overridden by DOCMIND_* variables "
            "(from your shell or .env) and take precedence over config.json.[/dim]"
        )


# --- REPL ------------------------------------------------------------------

_HELP = """[bold]Commands[/bold]
  add <path>              ingest a file/dir
  list                    show indexed docs
  remove <doc_id>         delete a doc
  ask <question>          answer with citations (auto local/remote)
  ask! <question>         force remote
  gaps [scope]            gap analysis (remote)
  critique [scope]        design review (remote)
  diagram <desc>          Mermaid diagram (remote)
  mindmap <topic>         Mermaid mind map (remote)
  provider [claude|kimi]  show/switch remote provider
  config                  show configuration
  help                    this help
  quit / exit             leave
"""


def _repl() -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory

    eng = _engine()
    console.print(
        Panel.fit(
            f"docmind — provider: [cyan]{eng.cfg.remote_provider}[/cyan] · "
            f"local: [cyan]{eng.cfg.local_chat_model}[/cyan]\nType [bold]help[/bold] for commands.",
            title="interactive",
        )
    )
    session: PromptSession = PromptSession(history=InMemoryHistory())

    while True:
        try:
            line = session.prompt("docmind> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        try:
            if cmd in ("quit", "exit"):
                break
            elif cmd == "help":
                console.print(_HELP)
            elif cmd == "add" and rest:
                eng.ingest_path(Path(rest), _log)
            elif cmd == "list":
                _repl_list(eng)
            elif cmd == "remove" and rest:
                eng.remove(rest, _log)
            elif cmd == "ask" and rest:
                _run_ask(eng, rest, None, False, False)
            elif cmd == "ask!" and rest:
                _run_ask(eng, rest, None, True, False)
            elif cmd == "gaps":
                _run_gaps(eng, rest or None, None, critique=False)
            elif cmd == "critique":
                _run_gaps(eng, rest or None, None, critique=True)
            elif cmd == "diagram" and rest:
                _run_diagram(eng, rest, False, None, mindmap=False)
            elif cmd == "mindmap" and rest:
                _run_diagram(eng, rest, False, None, mindmap=True)
            elif cmd == "provider":
                _repl_provider(eng, rest)
            elif cmd == "config":
                _show_config(eng.cfg)
            else:
                console.print("[dim]Unknown command. Type `help`.[/dim]")
        except Exception as e:  # keep the REPL alive on any task error
            console.print(f"[red]error:[/red] {e}")

    console.print("[dim]bye[/dim]")


def _repl_list(eng: Engine) -> None:
    docs = eng.manifest.all()
    if not docs:
        console.print("[dim]No documents indexed.[/dim]")
        return
    for d in docs:
        console.print(f"[cyan]{d.doc_id}[/cyan]  {d.source_path}  ({d.chunk_count} chunks, {d.diagram_count} diagrams)")


def _repl_provider(eng: Engine, rest: str) -> None:
    if not rest:
        console.print(f"provider: [cyan]{eng.cfg.remote_provider}[/cyan]")
        return
    if rest not in AVAILABLE_PROVIDERS:
        console.print(f"[red]unknown provider: {rest}[/red] (choose {', '.join(AVAILABLE_PROVIDERS)})")
        return
    eng.cfg.set("remote_provider", rest)
    console.print(f"[green]switched provider → {rest}[/green]")


if __name__ == "__main__":
    app()
