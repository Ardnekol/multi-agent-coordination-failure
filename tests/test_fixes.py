"""Tests for verdict parsing, honest run mode, fault outcome, and the Claude client."""
from types import SimpleNamespace as NS
import pytest
from app import llm_clients, mock_llm
from app.agents import AGENTS
from app.orchestrator import run_task, MissingKeysError
from app.protocol import parse_verdict
from app.failure_detector import detect

TASK = "Explain how solar panels work"


@pytest.mark.parametrize("text, expected", [
    ("APPROVED: fine", "APPROVED"),
    ("**APPROVED**: all steps covered", "APPROVED"),
    ("  \n approved? no. REJECTED: step 2 missing", "REJECTED"),   # lowercase prose ignored
    ("Verdict: APPROVED", "APPROVED"),
    ("REJECTED - cannot be APPROVED yet", "REJECTED"),             # first verdict wins
    ("## REJECTED\nStep 3 is wrong", "REJECTED"),
    ("Looks good to me!", None),
    ("", None),
])
def test_parse_verdict(text, expected):
    assert parse_verdict(text) == expected


def _msgs(reviewer_text):
    plan = "SUBTASKS:\n1. Research solar panels\n2. Draft solar panels\n3. Verify solar panels"
    exe = "Step 1: research on solar panels\nStep 2: draft of solar panels\nStep 3: verify solar panels"
    return [dict(step=1, round=1, role="planner", content=plan),
            dict(step=2, round=1, role="executor", content=exe),
            dict(step=3, round=1, role="reviewer", content=reviewer_text)]


def test_markdown_approval_is_not_a_deadlock():
    types = {f["type"] for f in detect(TASK, _msgs("**APPROVED**: all three steps are done."))}
    assert "NO_CONSENSUS_DEADLOCK" not in types and "PROTOCOL_VIOLATION" not in types


def test_missing_verdict_is_protocol_violation():
    types = {f["type"] for f in detect(TASK, _msgs("Looks good to me!"))}
    assert {"PROTOCOL_VIOLATION", "NO_CONSENSUS_DEADLOCK"} <= types


def test_orchestrator_accepts_formatted_approval(monkeypatch):
    monkeypatch.setitem(mock_llm.MOCK, "reviewer", lambda user: "**APPROVED** - looks complete")
    r = run_task(TASK)
    assert r["run"]["status"] == "approved" and r["run"]["score"] == 100


def _no_keys(monkeypatch):
    for name, (fn, _) in list(llm_clients._PROVIDERS.items()):
        monkeypatch.setitem(llm_clients._PROVIDERS, name, (fn, lambda: ""))


def test_live_without_keys_is_refused(monkeypatch):
    _no_keys(monkeypatch)
    with pytest.raises(MissingKeysError):
        run_task(TASK, live=True)


def test_live_with_fallback_is_marked_mixed(monkeypatch):
    _no_keys(monkeypatch)
    # only the planner's provider has a key (a stand-in "live" API that reuses the mock text)
    real_planner = mock_llm.MOCK["planner"]
    monkeypatch.setitem(AGENTS["planner"], "provider", "openai")
    monkeypatch.setitem(llm_clients._PROVIDERS, "openai",
                        (lambda system, user, model: real_planner(user), lambda: "fake-key"))
    r = run_task(TASK, live=True, allow_mock_fallback=True)
    assert r["run"]["mode"] == "mixed"
    assert [m["source"] for m in r["messages"]][:3] == ["live", "mock", "mock"]


def test_mock_run_is_marked_mock():
    r = run_task(TASK)
    assert r["run"]["mode"] == "mock" and r["run"]["fault_outcome"] == "n/a"


def test_fault_outcome_effective():
    assert run_task(TASK, fault="contradict")["run"]["fault_outcome"] == "effective"


def test_fault_outcome_resisted(monkeypatch):
    # an executor that ignores the injected line, like a robust real model would
    real_executor = mock_llm.MOCK["executor"]
    monkeypatch.setitem(mock_llm.MOCK, "executor",
                        lambda user: real_executor(user.replace("IGNORE the above", "")))
    r = run_task(TASK, fault="contradict")
    assert r["run"]["fault_outcome"] == "resisted" and r["run"]["score"] == 100


# ---- Claude-only default scenario ------------------------------------------------------------

def test_default_scenario_is_three_claude_models():
    assert {r: (a["provider"], a["model"]) for r, a in AGENTS.items()} == {
        "planner": ("claude", "claude-opus-5-5"),
        "executor": ("claude", "claude-sonnet-5-5"),
        "reviewer": ("claude", "claude-haiku-5-5"),
    }


def test_one_anthropic_key_is_enough_for_live(monkeypatch):
    _no_keys(monkeypatch)
    monkeypatch.setitem(llm_clients._PROVIDERS, "claude",
                        (lambda system, user, model: mock_llm.MOCK["planner"](user), lambda: "sk-ant-test"))
    assert llm_clients.missing_keys({a["provider"] for a in AGENTS.values()}) == []


def test_role_provider_can_be_switched(monkeypatch):
    from app import config
    monkeypatch.setenv("PLANNER_PROVIDER", "openai")
    monkeypatch.delenv("PLANNER_MODEL", raising=False)
    assert config._role("planner") == ("openai", config.OPENAI_MODEL)
    monkeypatch.setenv("PLANNER_PROVIDER", "claude")
    monkeypatch.setenv("PLANNER_MODEL", "claude-fable-5-2")
    assert config._role("planner") == ("claude", "claude-fable-5-2")
    monkeypatch.setenv("PLANNER_PROVIDER", "mistral")
    with pytest.raises(ValueError):
        config._role("planner")


def test_claude_text_skips_thinking_blocks():
    resp = NS(stop_reason="end_turn", content=[
        NS(type="thinking", thinking=""),                     # thinking comes first on current models
        NS(type="text", text="APPROVED: all three steps "),
        NS(type="text", text="are covered."),
    ])
    assert llm_clients.claude_text(resp) == "APPROVED: all three steps are covered."


def test_claude_refusal_becomes_agent_crash(monkeypatch):
    refused = NS(stop_reason="refusal", stop_details=NS(category="cyber"), content=[])
    with pytest.raises(llm_clients.AgentRefused, match="cyber"):
        llm_clients.claude_text(refused)

    def refusing_api(system, user, model):
        return llm_clients.claude_text(refused)
    monkeypatch.setitem(llm_clients._PROVIDERS, "claude", (refusing_api, lambda: "sk-ant-test"))
    text, _, source = llm_clients.call("claude", "claude-haiku-5-5", "reviewer", "sys", "user", live=True)
    assert source == "error" and text.startswith("[AGENT_ERROR] claude: model refused")
