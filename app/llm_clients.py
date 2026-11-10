"""One `call()` function in front of three LLM providers, with a mock fallback.

The orchestrator never talks to a provider SDK directly. It calls
`call(provider, role, system, user, live)` and gets back (text, latency_ms, source):

* live=False                -> the deterministic mock agent for `role` answers (mock_llm.py)
* live=True and key present -> the real provider API answers
* live=True and key missing -> the mock answers, and source says 'mock'. The orchestrator
                               refuses this case unless allow_mock_fallback is set.

SDKs are imported inside each wrapper, so mock mode works even when the provider packages
are not installed.
"""
import time
from . import config, mock_llm


def _openai(system, user):
    """Planner backend: OpenAI Chat Completions."""
    from openai import OpenAI
    r = OpenAI(api_key=config.OPENAI_API_KEY).chat.completions.create(
        model=config.OPENAI_MODEL, temperature=0.3,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content


def _gemini(system, user):
    """Executor backend: Google Gemini via the google-genai SDK."""
    from google import genai
    from google.genai import types
    r = genai.Client(api_key=config.GEMINI_API_KEY).models.generate_content(
        model=config.GEMINI_MODEL, contents=user,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.3))
    return r.text


def _claude(system, user):
    """Reviewer backend: Anthropic Messages API."""
    import anthropic
    r = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY).messages.create(
        model=config.CLAUDE_MODEL, max_tokens=800, system=system,
        messages=[{"role": "user", "content": user}])
    return r.content[0].text


# provider name -> (function that calls the API, function that returns the current key).
# The key is looked up through a lambda (not copied) so tests can swap entries at runtime.
_PROVIDERS = {
    "openai": (_openai, lambda: config.OPENAI_API_KEY),
    "gemini": (_gemini, lambda: config.GEMINI_API_KEY),
    "claude": (_claude, lambda: config.ANTHROPIC_API_KEY),
}


def missing_keys(providers):
    """Providers in `providers` that have no API key configured."""
    return sorted({p for p in providers if not _PROVIDERS[p][1]()})


def call(provider: str, role: str, system: str, user: str, live: bool):
    """Returns (text, latency_ms, source).

    source is 'live' (real API answered), 'error' (real API was called and failed) or
    'mock' (live=False, or the provider has no key). Callers use it so a run that silently
    fell back to the mock is never reported as live.

    This function never raises: an exception becomes the text "[AGENT_ERROR] <provider>: ..."
    so the run continues and the detector records it as AGENT_CRASH.
    """
    start = time.time()
    fn, key = _PROVIDERS[provider]
    use_live = live and bool(key())
    try:
        if use_live:
            text, source = fn(system, user), "live"
        else:
            text, source = mock_llm.MOCK[role](user), "mock"
    except Exception as e:                      # an API crash is itself a coordination failure
        text, source = f"[AGENT_ERROR] {provider}: {e}", "error" if use_live else "mock"
    # `text or ""`: some SDKs return None for an empty or blocked response
    return text or "", int((time.time() - start) * 1000), source
