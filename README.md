# docmind

A **local-first document intelligence agent**. Ingest a personal corpus of
process/architecture docs (`.pdf`, `.docx`, `.txt`, `.md` — diagrams included),
ask questions across them with citations, get a senior-tech-management gap
analysis, and generate Mermaid diagrams / mind maps. Use it from the **CLI**, an
interactive **terminal REPL**, or a **web UI** (`docmind ui`) with saved chat
history.

- **Local-first**: parsing, embeddings, retrieval, diagram understanding, and
  routine Q&A all run on your machine via [Ollama]
  (`qwen3-embedding`, `qwen3:14b`, `qwen2.5vl` for diagrams).
- **Escalates only when it helps**: gap analysis, architecture critique, and
  diagram generation auto-route to a **remote provider**; everything else stays
  local.
- **Swappable provider**: Claude (Anthropic) or Kimi K3 (Moonshot), by config,
  `--provider` flag, or `provider` in the REPL. Default: `claude-opus-4-8`.
- **Cost-aware**: every remote call prints an estimated $ first, respects a
  confirmation threshold and a per-session budget cap, and prompt-caches the
  shared corpus context.

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| macOS or Linux | — | `setup.sh` targets a Unix shell (`bash`). |
| **Homebrew** | any | Only needed to bootstrap `mise` if it isn't installed. https://brew.sh |
| **mise** | ≥ 2024 | Runtime/venv manager; `setup.sh` installs it via brew if missing. https://mise.jdx.dev |
| **Python** | **3.13** | Pinned in `mise.toml` and installed by mise (package itself runs on ≥ 3.11). |
| **Node.js** | **24** | Installed by mise; used only for optional Mermaid rendering (`--render` via `npx @mermaid-js/mermaid-cli`). |
| **Ollama** | ≥ 0.3, running | Provides all local models. Install from https://ollama.com and ensure the daemon is running (`ollama serve`). |

### Ollama models (local, ~20 GB total)

`setup.sh` pulls the vision model and checks the other two:

| Model | Purpose | Size |
|---|---|---|
| `qwen3:14b` | Local chat / routine Q&A | ~9.3 GB |
| `qwen3-embedding` | Embeddings for retrieval | ~4.7 GB |
| `qwen2.5vl` | Local vision (diagram descriptions) | ~6 GB |

Pull manually if needed: `ollama pull qwen3:14b qwen3-embedding qwen2.5vl`.

### Python dependencies

Installed into the project `.venv` by `setup.sh` (from `pyproject.toml`): `typer`,
`rich`, `prompt_toolkit`, `pymupdf`, `python-docx`, `lancedb`, `pyarrow`,
`ollama`, `anthropic`, `openai`. The web UI adds `streamlit` (installed via the
`ui` extra — `setup.sh` includes it; or `pip install -e ".[ui]"`).

### Optional — remote providers

Only for `gaps` / `critique` / `diagram` / escalated `ask`. Without them docmind
runs fully local.

- **Claude**: `ANTHROPIC_API_KEY` (or `ant auth login`)
- **Kimi K3**: `KIMI_API_KEY`

## Setup

```bash
./setup.sh          # installs mise (via brew) → python 3.13 + node → .venv → deps → ollama pull
```

The environment is managed by **mise** (`mise.toml` auto-creates/activates
`.venv`). Re-running `setup.sh` is safe.

Remote-provider credentials are optional (only needed for `gaps` / `critique` /
`diagram` / escalated `ask`). The easiest way is a **`.env` file** — `setup.sh`
creates one from `.env.example`; just fill in the keys:

```bash
# .env  (git-ignored; loaded automatically on startup)
ANTHROPIC_API_KEY=sk-ant-...
KIMI_API_KEY=sk-...
```

Or export them in the shell (exported vars override `.env`):

```bash
export ANTHROPIC_API_KEY=...   # or: ant auth login
export KIMI_API_KEY=...
```

Without any key, docmind runs fully local.

### Running the `docmind` command

`docmind` is installed into the project `.venv`. Use whichever you prefer:

```bash
# A) activate mise once, then `docmind` works inside the repo (auto-venv)
eval "$(mise activate zsh)"   # or: bash — add to ~/.zshrc/~/.bashrc to make permanent
cd /path/to/reggae            # mise auto-activates .venv here
docmind --help

# B) no shell setup — prefix any command with `mise exec --`
mise exec -- docmind --help

# C) call the venv binary directly
.venv/bin/docmind --help
```

### Quick start

```bash
./setup.sh                         # one-time environment + deps
docmind add ./docs                 # ingest your documents
docmind ui                         # launch the web UI (http://localhost:8501)
# ...or from the terminal:
docmind ask "what triggers onboarding?"
```

## Usage

```bash
docmind add ./docs                 # ingest a file or directory
docmind list                       # show indexed docs
docmind ask "what triggers onboarding?"
docmind gaps --scope "order flow"  # gap analysis (remote, falls back to local)
docmind critique                   # design review (remote, falls back to local)
docmind diagram "end-to-end order flow" --render   # Mermaid → SVG
docmind mindmap "incident response"
docmind mindmap --local --freeform "MO system: ingestion, engine, reporting"
docmind remove <doc_id>            # clean removal (chunks + images)
docmind config show                # view/change settings
docmind repl                       # interactive terminal session
docmind ui                         # web UI with saved chat history (see below)
```

`--provider claude|kimi` on any remote command overrides the active provider for
that call; `docmind config set remote_provider kimi` changes the default.

### Local vs. remote

`ask` auto-routes (lookups stay local; analytical questions escalate). `gaps`,
`critique`, `diagram`, and `mindmap` prefer the remote provider but **fall back
to the local model automatically** when no API key is set — so everything works
offline. Control it explicitly:

- `--local` — force the local model (`qwen3:14b`), no key needed, `$0`.
- `--remote` (on `ask`) / omit `--local` — use the remote provider.
- `--freeform` (on `diagram`/`mindmap`) — build from your text alone, ignoring
  the corpus (handy when the topic isn't in your documents).

## Web UI

```bash
docmind ui                 # launches Streamlit at http://localhost:8501
docmind ui --port 8600     # custom port
```

The UI is a thin front-end over the same engine — everything the CLI does, in a
browser:

- **Chat** with streamed answers, a per-turn badge (route / model / cost) and a
  Sources panel; a Route toggle (Auto/Local/Remote) and provider selector.
- **Documents** (sidebar): drag-drop upload → ingest, list with chunk/diagram
  counts, remove, and **Re-index all**.
- **Slash commands** (typed in the chat box, results land in the transcript):
  `/gaps [scope]`, `/critique [scope]`, `/diagram <desc>`, `/mindmap <topic>`,
  `/help`. Add `--freeform` to `/diagram` / `/mindmap` to build from your text,
  ignoring the corpus. Diagrams render inline as SVG (rendered locally via
  mermaid-cli — no internet needed) with `.mmd`/`.svg` downloads. The **Route**
  toggle (Auto/Local/Remote) controls where they run; with no key they fall
  back to local.
- **Chat history** (sidebar): sessions are saved to `docmind/data/chats/` as
  JSON — resume after a refresh or restart, rename, delete, or export to
  Markdown.
- **Config** (sidebar): edit models, provider, `top_k`, chunking, and the cost
  guard live (chunking changes prompt a **Re-index all**).

The cost guard applies here too: remote calls show an estimate first and respect
the session budget cap. Requires the `ui` extra (see Setup).

## Configuration

**Two kinds of config, kept separate on purpose:**

| What | Where | How |
|---|---|---|
| **Secrets** (API keys) | `.env` or shell env | `ANTHROPIC_API_KEY`, `KIMI_API_KEY` |
| **Settings** (models, provider, tuning, cost guard) | `docmind/data/config.json` | `docmind config set …` |

So configuring "use Claude model X" is normally: put the **key** in `.env`, set
the **model** with `docmind config set claude_model X`. If you'd rather keep
everything in one place, any setting can also be overridden from the environment
(e.g. in `.env`) with `DOCMIND_<KEY>`:

```bash
# .env — settings overrides (optional); take precedence over config.json
DOCMIND_REMOTE_PROVIDER=claude
DOCMIND_CLAUDE_MODEL=claude-opus-4-8
DOCMIND_TOP_K=12
```

**Precedence:** `DOCMIND_*` env var → `config.json` → built-in default.
`docmind config show` marks which keys are currently coming from the environment.

Settings persist to `docmind/data/config.json` (created on first run). Manage
them with:

```bash
docmind config show                 # list every setting and its value
docmind config get <key>            # print one value
docmind config set <key> <value>    # change one value (saved immediately)
```

### All settings

| Key | Default | What it does |
|---|---|---|
| `ollama_host` | `http://localhost:11434` | Ollama endpoint. |
| `embed_model` | `qwen3-embedding` | Local embedding model. **Changing requires `add --force` re-index.** |
| `local_chat_model` | `qwen3:14b` | Local model for routine Q&A. |
| `vision_model` | `qwen2.5vl` | Local model that describes diagrams at ingest. |
| `remote_provider` | `claude` | Active remote provider: `claude` or `kimi`. |
| `claude_model` | `claude-opus-4-8` | Claude model for escalated `ask`. |
| `claude_heavy_model` | `claude-opus-4-8` | Claude model for `gaps` / `critique`. |
| `claude_effort` | `high` | Effort for supported Claude models (`low`/`medium`/`high`/`xhigh`/`max`). |
| `kimi_base_url` | `https://api.moonshot.ai/v1` | Moonshot OpenAI-compatible endpoint. |
| `kimi_model` | `kimi-k3-turbo-preview` | Kimi model for escalated `ask`. |
| `kimi_heavy_model` | `kimi-k3-turbo-preview` | Kimi model for `gaps` / `critique`. |
| `top_k` | `8` | Chunks retrieved per query (applies immediately). |
| `chunk_tokens` | `800` | Target chunk size. **Ingest-time — needs `add --force`.** |
| `chunk_overlap` | `120` | Overlap between chunks. **Ingest-time — needs `add --force`.** |
| `budget_cap` | `5.0` | Per-session remote spend cap (USD). |
| `confirm_threshold` | `0.10` | Prompt before any remote call estimated above this (USD). |

### Common recipes

```bash
# Tune chunking (re-index required to apply to existing docs)
docmind config set chunk_tokens 500
docmind config set chunk_overlap 80
docmind add ./docs --force

# Retrieve more context per query (no re-index)
docmind config set top_k 12

# Switch the local Q&A model (pull it first)
ollama pull qwen3:30b-a3b
docmind config set local_chat_model qwen3:30b-a3b

# Switch remote provider + models
docmind config set remote_provider kimi
docmind config set claude_heavy_model claude-sonnet-5   # cheaper for gaps/critique

# Cost guard
docmind config set budget_cap 2.0
docmind config set confirm_threshold 0.25
```

> **Which changes need a re-index?** Only `embed_model`, `chunk_tokens`, and
> `chunk_overlap` affect stored data — after changing any of them, run
> `docmind add <path> --force`. Everything else (models, provider, `top_k`,
> cost guard) takes effect on the next command.

## How it works

`ingest → parse → describe diagrams (local vision) → chunk → embed (Ollama) →
LanceDB`. Queries retrieve top-k chunks (with provenance) and a router sends
them to the local model or, for heavyweight tasks, the active remote provider.
Diagrams are described once at ingest and stored as searchable chunks, so their
semantics are queryable without re-running vision. Removing a document is a
single predicate delete on `doc_id`, so the index never accumulates stragglers.

See `docmind/` for the module layout. All local state lives under
`docmind/data/` (created on first run, git-ignored):

| Path | Contents |
|---|---|
| `config.json` | Settings (see Configuration). |
| `lancedb/` | Vector store (chunks + embeddings + metadata). |
| `manifest.json` | Indexed-document registry. |
| `images/<doc_id>/` | Diagram images extracted at ingest. |
| `diagrams/` | Generated Mermaid (`.mmd` / `.md` / `.svg`). |
| `chats/` | Saved UI chat sessions (JSON). |
| `uploads/` | Files uploaded through the web UI. |

[Ollama]: https://ollama.com
