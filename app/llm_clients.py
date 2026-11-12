"""One `call()` function in front of three LLM providers, with a mock fallback.

The orchestrator never talks to a provider SDK directly. It calls
`call(provider, model, role, system, user, live)` and gets back (text, latency_ms, source):

* live=False                -> the deterministic mock agent for `role` answers (mock_llm.py)
* live=True and key present -> the real provider API answers
* live=True and key missing -> the mock answers, and source says 'mock'. The orchestrator
                               refuses this case unless allow_mock_fallback is set.

SDKs are imported inside each wrapper, so mock mode works even when the provider packages
are not installed.
"""
import time
from . import config, mock_llm

# Claude models whose API supports server-side refusal fallbacks (Haiku has none).
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1")


class AgentRefused(RuntimeError):
    """The model declined the request (stop_reason 'refusal')."""


def _openai(system, user, model):
    """OpenAI Chat Completions."""
    from openai import OpenAI
    r = OpenAI(api_key=config.OPENAI_API_KEY).chat.completions.create(
        model=model, temperature=0.3,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content


def _gemini(system, user, model):
    """Google Gemini via the google-genai SDK."""
    from google import genai
    from google.genai import types
    r = genai.Client(api_key=config.GEMINI_API_KEY).models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.3))
    return r.text


def claude_text(response):
    """The answer text of a Claude response.

    Current Claude models think by default, so `content` can start with `thinking` blocks
    (empty text unless summaries are requested) before the `text` blocks. Only text blocks
    are the answer; reading content[0] blindly would return the wrong block.
    Raises AgentRefused when the model declined, so the run records it as AGENT_CRASH.
    """
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise AgentRefused(f"model refused the request (category: {category or 'unspecified'})")
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _claude(system, user, model):
    """Anthropic Messages API.

    - No temperature: current Claude models reject non-default sampling settings.
    - max_tokens 16000: thinking tokens count toward this limit, so a small value can cut the
      answer off mid-way. Short answers still only use what they need.
    - CLAUDE_EFFORT (optional) sets output_config.effort; unset = the model's default.
    - CLAUDE_FALLBACK: on Opus/Sonnet, a refusal is retried on a fallback model chosen by the
      API ("default" routing), so a run is not lost to an over-cautious classifier.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    params = dict(model=model, max_tokens=16000, system=system,
                  messages=[{"role": "user", "content": user}])
    if config.CLAUDE_EFFORT:
        params["output_config"] = {"effort": config.CLAUDE_EFFORT}
    if config.CLAUDE_FALLBACK and model in _FALLBACK_MODELS:
        r = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                        fallbacks="default", **params)
    else:
        r = client.messages.create(**params)
    return claude_text(r)


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


def call(provider: str, model: str, role: str, system: str, user: str, live: bool):
    """Returns (text, latency_ms, source).

    source is 'live' (real API answered), 'error' (real API was called and failed) or
    'mock' (live=False, or the provider has no key). Callers use it so a run that silently
    fell back to the mock is never reported as live.

    This function never raises: an exception (including a refusal) becomes the text
    "[AGENT_ERROR] <provider>: ..." so the run continues and the detector records AGENT_CRASH.
    """
    start = time.time()
    fn, key = _PROVIDERS[provider]
    use_live = live and bool(key())
    try:
        if use_live:
            text, source = fn(system, user, model), "live"
        else:
            text, source = mock_llm.MOCK[role](user), "mock"
    except Exception as e:                      # an API crash is itself a coordination failure
        text, source = f"[AGENT_ERROR] {provider}: {e}", "error" if use_live else "mock"
    # `text or ""`: some SDKs return None for an empty or blocked response
    return text or "", int((time.time() - start) * 1000), source
