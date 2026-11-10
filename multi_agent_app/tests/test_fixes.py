"""Tests for verdict parsing, honest run mode, and fault outcome."""
import pytest
from app import llm_clients, mock_llm
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
    real_planner = mock_llm.MOCK["planner"]
    monkeypatch.setitem(llm_clients._PROVIDERS, "openai",
                        (lambda system, user: real_planner(user), lambda: "fake-key"))
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
