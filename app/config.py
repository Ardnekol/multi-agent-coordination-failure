"""Runtime configuration, read once from environment variables (and `.env` if present).

Every setting has a safe default, so the app starts with no configuration at all and runs the
mock agents. API keys are only needed for live mode (see `llm_clients.py`).

| Variable            | Used for                                  | Default              |
|---------------------|-------------------------------------------|----------------------|
| OPENAI_API_KEY      | Planner agent in live mode                | "" (not set)         |
| GEMINI_API_KEY      | Executor agent in live mode               | "" (not set)         |
| ANTHROPIC_API_KEY   | Reviewer agent in live mode               | "" (not set)         |
| OPENAI_MODEL        | Model name sent to OpenAI                 | gpt-4o-mini          |
| GEMINI_MODEL        | Model name sent to Gemini                 | gemini-2.0-flash     |
| CLAUDE_MODEL        | Model name sent to Anthropic              | claude-sonnet-5-5    |
| DB_PATH             | SQLite file that stores every run         | coordination.db      |

Values are read at import time, so changing `.env` needs a server restart. The tests rely on
this: `tests/conftest.py` sets DB_PATH *before* importing `app`, which points them at a
throwaway database.
"""
import os

# python-dotenv is optional: without it, only real environment variables are used.
try:
    from dotenv import load_dotenv
    load_dotenv()                     # copies KEY=value lines from ./.env into os.environ
except ImportError:
    pass

# --- Provider credentials (empty string means "not configured") -------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Model names: providers rename/retire models, so these are overridable -------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

# --- Storage ----------------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "coordination.db")
