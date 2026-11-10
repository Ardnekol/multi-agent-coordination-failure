"""Mock experiment that uses the README's own problem statement as the agents' task.

Run with `pytest -q -s tests/test_readme_task.py` to also print the results table.
"""
import re
from pathlib import Path
import pytest
from app.orchestrator import run_task

README = Path(__file__).resolve().parents[1] / "README.md"


def problem_statement():
    text = README.read_text(encoding="utf-8")
    m = re.search(r"^## 1\. Problem statement\s*\n(.*?)(?=^## )", text, re.S | re.M)
    assert m, "README.md has no '## 1. Problem statement' section"
    return m.group(1).strip()


TASK = problem_statement()

# (fault, fault_rounds) -> expected (status, score, fault_outcome, failure types)
# These are what the deterministic mock agents must produce; a change here means the
# orchestrator, mock or detector changed behaviour.
EXPECTED = {
    ("none", 1):           ("approved", 100, "n/a", set()),
    ("drop", 1):           ("approved", 85, "effective", {"EMPTY_OR_TRUNCATED_HANDOFF"}),
    ("truncate", 1):       ("approved", 85, "effective", {"INSTRUCTION_DROPPED"}),
    ("contradict", 1):     ("approved", 65, "effective", {"TASK_DRIFT", "INSTRUCTION_DROPPED"}),
    ("role_confusion", 1): ("approved", 65, "effective", {"EMPTY_OR_TRUNCATED_HANDOFF", "ROLE_CONFUSION"}),
    ("drop", 3):           ("deadlock", 0, "effective",
                            {"EMPTY_OR_TRUNCATED_HANDOFF", "REPETITION_LOOP",
                             "REVIEW_PING_PONG", "NO_CONSENSUS_DEADLOCK"}),
}


def test_problem_statement_is_a_valid_task():
    assert 5 <= len(TASK) <= 500          # same limits as the API's RunRequest
    assert "\n" in TASK                   # multi-line on purpose: exercises the mock planner fix


def test_planner_keeps_the_whole_multiline_task():
    plan = run_task(TASK)["messages"][0]["content"]
    assert "quantifiable" in plan          # last word of the statement, on its last line


@pytest.mark.parametrize("fault, rounds", list(EXPECTED))
def test_readme_task_under_each_fault(fault, rounds):
    r = run_task(TASK, fault=fault, fault_rounds=rounds)
    run, types = r["run"], {f["type"] for f in r["failures"]}
    print(f"\n{fault:>14} x{rounds}  status={run['status']:<8} score={run['score']:>3}  "
          f"fault={run['fault_outcome']:<9} mode={run['mode']}  failures={sorted(types)}")
    status, score, outcome, expected_types = EXPECTED[(fault, rounds)]
    assert run["mode"] == "mock"
    assert (run["status"], run["score"], run["fault_outcome"]) == (status, score, outcome)
    assert types == expected_types
