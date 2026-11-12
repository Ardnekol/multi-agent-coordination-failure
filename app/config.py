"""Runtime configuration, read once from environment variables (and `.env` if present).

Every setting has a safe default, so the app starts with no configuration at all and runs the
mock agents. API keys are only needed for live mode (see `llm_clients.py`).

Default scenario: all three agents run on Claude, each on a different model size
(Opus plans, Sonnet executes, Haiku reviews), so one Anthropic key is enough for live runs.
Any role can be moved to another provider with <ROLE>_PROVIDER / <ROLE>_MODEL.

| Variable            | Used for                                         | Default              |
|---------------------|--------------------------------------------------|----------------------|
| ANTHROPIC_API_KEY   | every agent whose provider is "claude"           | "" (not set)         |
| OPENAI_API_KEY      | agents whose provider is "openai"                | "" (not set)         |
| GEMINI_API_KEY      | agents whose provider is "gemini"                | "" (not set)         |
| PLANNER_PROVIDER    | claude / openai / gemini                         | claude               |
| PLANNER_MODEL       | model name for the planner                       | claude-opus-5-5      |
| EXECUTOR_PROVIDER   | claude / openai / gemini                         | claude               |
| EXECUTOR_MODEL      | model name for the executor                      | claude-sonnet-5-5    |
| REVIEWER_PROVIDER   | claude / openai / gemini                         | claude               |
| REVIEWER_MODEL      | model name for the reviewer                      | claude-haiku-5-5     |
| OPENAI_MODEL        | default model when a role uses openai            | gpt-4o-mini          |
| GEMINI_MODEL        | default model when a role uses gemini            | gemini-2.0-flash     |
| CLAUDE_EFFORT       | optional effort for Claude calls (low...max)     | "" (model default)   |
| CLAUDE_FALLBACK     | "1" = on a refusal, let the API retry on a       | 1                    |
|                     | fallback model (Opus/Sonnet only)                |                      |
| DB_PATH             | SQLite file that stores every run                | coordination.db      |

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

# --- Provider default models: used when a role is switched to that provider without a model ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# --- Claude request options -------------------------------------------------------------------
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "").strip().lower()      # "" = let the model decide
CLAUDE_FALLBACK = os.getenv("CLAUDE_FALLBACK", "1").strip() not in ("0", "false", "no", "")

# --- Which provider and model each agent role uses --------------------------------------------
PROVIDERS = ("claude", "openai", "gemini")
_CLAUDE_DEFAULTS = {               # different sizes on purpose: capability gaps cause failures
    "planner": "claude-opus-5-5",
    "executor": "claude-sonnet-5-5",
    "reviewer": "claude-haiku-5-5",
}


def _role(role):
    """(provider, model) for one role, from <ROLE>_PROVIDER / <ROLE>_MODEL."""
    provider = os.getenv(f"{role.upper()}_PROVIDER", "claude").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"{role.upper()}_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
    default = {"claude": _CLAUDE_DEFAULTS[role], "openai": OPENAI_MODEL, "gemini": GEMINI_MODEL}[provider]
    return provider, os.getenv(f"{role.upper()}_MODEL", "").strip() or default


ROLE_MODELS = {role: _role(role) for role in ("planner", "executor", "reviewer")}

# --- Storage ----------------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "coordination.db")
