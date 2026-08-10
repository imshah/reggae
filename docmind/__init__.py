"""docmind — a local-first document intelligence agent."""

from docmind.env import load_env

# Load .env (API keys) before any provider reads os.environ.
load_env()

__version__ = "0.1.0"
