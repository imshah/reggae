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
- **Kimi (Moonshot)**: `KIMI_API_KEY`

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

### Enable Kimi (Moonshot)

1. Put your key in `.env`: `KIMI_API_KEY=<key from platform.kimi.ai>`.
2. **Top up the account** — a new key returns errors until it has a minimum balance.
3. Select the provider (persists to `config.json`):

   ```bash
   docmind config set remote_provider kimi   # default for all remote calls
   docmind ask "..." --provider kimi         # or override per call
   ```

Defaults: `kimi-k2.6` for escalated `ask`, `kimi-k3` for `gaps`/`critique`, against
`https://api.moonshot.ai/v1`. Override any of these with `config set` (e.g.
`docmind config set kimi_model kimi-k2.7-code`). Model availability varies by account
— confirm the ids yours exposes with
`curl -s https://api.moonshot.ai/v1/models -H "Authorization: Bearer $KIMI_API_KEY"`.

> **Precedence:** if `DOCMIND_REMOTE_PROVIDER` is set in your shell or `.env`, it
> overrides whatever `config set` persists.

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
docmind add ./docs                 # ingest into the active group
docmind add ./docs --group ops     # ingest into a specific group
docmind add ./docs -g ops -g qa    # ingest into several groups at once
docmind list                       # docs in the active group  (--all for every group)
docmind ask "what triggers onboarding?"            # scoped to the active group
docmind ask "..." --group ops                      # scope one query to a group
docmind ask "..." --all-groups                     # search across all groups
docmind gaps --scope "order flow"  # gap analysis (remote, falls back to local)
docmind critique                   # design review (remote, falls back to local)
docmind diagram "end-to-end order flow" --render   # Mermaid → SVG
docmind mindmap "incident response"
docmind mindmap --local --freeform "MO system: ingestion, engine, reporting"
docmind remove <doc_id>            # clean removal (chunks + images)
docmind group                        # show active group + all groups with counts
docmind group use ops                # set the active group (scopes queries/ingest)
docmind group add <doc_id> projX     # add a document to a group (keeps existing ones)
docmind group remove-doc <doc_id> projX  # detach a document from a group
docmind group remove projX           # detach every doc from a group (docs are kept)
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
- `--group <name>` / `--all-groups` (on `ask`/`gaps`/`critique`/`diagram`/
  `mindmap`) — scope one call to a group, or search across all groups.

## Groups

Organise documents into **groups** (collections) so unrelated corpora never
pollute each other's answers — e.g. `onboarding` vs `order-system` vs a client
project. **A document can belong to several groups** (default: `default`), so the
same doc can be surfaced under multiple collections without duplicating it.
Queries are **scoped to the active group** unless you say otherwise.

```bash
docmind add ./client-x --group clientX   # ingest into a group (creates it if new)
docmind add ./shared -g clientX -g ops   # ingest into several groups at once
docmind group                            # show the active group + all groups (with counts)
docmind group use clientX                # switch the active group — scopes future queries/ingest
docmind ask "what's the SLA?"            # answered only from clientX docs
docmind ask "..." --group ops            # override scope for one call
docmind ask "..." --all-groups           # search every group at once
docmind group add <doc_id> ops           # add a doc to another group (keeps existing ones)
docmind group remove-doc <doc_id> ops    # detach a doc from a group
docmind group remove clientX             # detach every doc from a group (docs are kept)
```

Membership is additive: re-`add`-ing an already-indexed file with a new `--group`
**adds** that group rather than moving the doc. Removing a group only **detaches**
it — no documents are deleted, and a doc left with no groups falls back to
`default`.

Existing indexes migrate automatically: documents indexed before groups existed
land in `default` (no re-index). In the terminal REPL, `group` shows/lists and
`group <name>` switches the active group; append `--all` to a command to span all
groups.

## Web UI

```bash
docmind ui                 # launches Streamlit at http://localhost:8501
docmind ui --port 8600     # custom port
```

The UI is a thin front-end over the same engine — everything the CLI does, in a
browser:

- **Chat** with streamed answers, a per-turn badge (route / model / cost) and a
  Sources panel; a Route toggle (Auto/Local/Remote) and provider selector.
- **Groups** (sidebar): a **Group** selector scopes what you view and query
  ("All groups" searches everything). At the uploader, an **"Ingest into group"**
  field tags new files — type a new name to create a group on the fly. Answers,
  slash commands, and analysis all stay within the selected group.
- **Documents** (sidebar): drag-drop upload → ingest, list (filtered by the
  selected group) with chunk/diagram counts, a per-doc **Edit groups**
  multiselect (add/remove memberships, or type a new group) and remove, and
  **Re-index all** (preserving each doc's groups).
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
| `kimi_model` | `kimi-k2.6` | Kimi model for escalated `ask`. |
| `kimi_heavy_model` | `kimi-k3` | Kimi model for `gaps` / `critique`. |
| `top_k` | `8` | Chunks retrieved per query (applies immediately). |
| `active_group` | `default` | Group scope for queries/ingest (env: `DOCMIND_ACTIVE_GROUP`). |
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
