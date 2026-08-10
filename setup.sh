#!/usr/bin/env bash
# Idempotent bootstrap for docmind. Safe to re-run.
# Manages the environment with mise (installs it via brew if missing),
# provisions Python 3.13 + Node, creates a project-local .venv, installs
# dependencies, and pulls the local vision model into Ollama.
set -euo pipefail

cd "$(dirname "$0")"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN\033[0m %s\n' "$*"; }

# --- 1. mise ---------------------------------------------------------------
if ! command -v mise >/dev/null 2>&1; then
  say "mise not found — installing via brew"
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required to bootstrap mise. Install from https://brew.sh" >&2
    exit 1
  fi
  brew install mise
  cat <<'EOF'

  mise installed. To activate it automatically in new shells, add ONE of:
    bash:  echo 'eval "$(mise activate bash)"' >> ~/.bashrc
    zsh:   echo 'eval "$(mise activate zsh)"'  >> ~/.zshrc
  Then restart your shell. For this run, setup.sh uses `mise exec` directly.

EOF
else
  say "mise present: $(mise --version)"
fi

# --- 2. runtimes + venv ----------------------------------------------------
say "Trusting mise config and installing runtimes (python 3.13, node)"
mise trust >/dev/null
mise install

# mise.toml auto-creates .venv; ensure it exists before installing into it.
say "Ensuring project virtualenv (.venv) exists"
mise exec -- python -m venv .venv 2>/dev/null || true

# --- 3. python deps --------------------------------------------------------
say "Installing Python dependencies (editable, with UI extra)"
mise exec -- .venv/bin/python -m pip install --upgrade pip >/dev/null
mise exec -- .venv/bin/python -m pip install -e ".[ui]"

# --- 3b. .env for API keys -------------------------------------------------
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  say "Created .env from template — add API keys there (optional; local works without)"
fi

# --- 4. ollama models ------------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
  have_model() { ollama list 2>/dev/null | awk '{print $1}' | grep -q "^$1"; }

  for m in qwen3:14b qwen3-embedding; do
    if have_model "$m"; then
      say "Ollama model present: $m"
    else
      warn "Ollama model missing: $m (expected already pulled) — run: ollama pull $m"
    fi
  done

  if have_model "qwen2.5vl"; then
    say "Ollama vision model present: qwen2.5vl"
  else
    say "Pulling local vision model: qwen2.5vl (this may take a while)"
    ollama pull qwen2.5vl
  fi
else
  warn "ollama not found on PATH — install it and pull qwen3:14b, qwen3-embedding, qwen2.5vl"
fi

# --- 5. done ---------------------------------------------------------------
cat <<'EOF'

Setup complete.

Run the agent:
  mise exec -- docmind --help        # or, with mise activated: docmind --help
  docmind add ./path/to/docs
  docmind repl                       # interactive terminal session
  docmind ui                         # web UI (Streamlit)

Remote-provider credentials (only needed for gaps/critique/diagram or escalated ask):
  Claude:  export ANTHROPIC_API_KEY=...   (or `ant auth login`)
  Kimi:    export KIMI_API_KEY=...
Without them, docmind runs fully local (qwen3:14b + qwen3-embedding).
EOF
