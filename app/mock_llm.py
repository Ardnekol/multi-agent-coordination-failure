"""Deterministic fake agents so the project runs without API keys (demo + tests).

Each function takes the same user prompt a real model would get and returns a canned answer
in the protocol format from agents.py. The same input always gives the same output, which
makes the tests exact.

Important for interpreting results: the mock executor recognises the injected fault text
("You are the Reviewer", "IGNORE the above") and deliberately fails. Mock runs therefore show
that the pipeline and detector work end to end; they do NOT show how real models react to
faults. Use live mode for that.
"""
import re


def planner(user: str) -> str:
    """Return a 3-subtask plan that repeats the task; adds 'REVISED PLAN' after feedback."""
    # tasks can span several lines (e.g. a pasted paragraph); keep all of it, on one line
    task = re.search(r"Task: (.*?)(?:\nReviewer feedback|\Z)", user, re.S).group(1)
    task = " ".join(task.split())
    rev = "REVISED PLAN\n" if "feedback" in user.lower() else ""
    return (f"{rev}SUBTASKS:\n"
            f"1. Research background information about {task}\n"
            f"2. Draft the main solution for {task}\n"
            f"3. Verify and summarise the final result for {task}")


def executor(user: str) -> str:
    """Return one 'Step N' line per numbered plan line it can see.

    Simulated failure behaviour:
      role_confusion fault -> answers "APPROVED" as if it were the reviewer
      contradict fault     -> follows the injected instruction and writes a poem
      drop fault           -> no numbered lines arrive, so "No plan received."
      truncate fault       -> only the subtasks that survived the cut get a Step line
    """
    if "You are the Reviewer" in user:                      # role-confusion fault
        return "APPROVED"
    if "IGNORE the above" in user:                          # contradiction fault
        return "Here is a poem about the weather: sunny skies and a gentle breeze all day long."
    steps = re.findall(r"^\d+\.\s*(.+)$", user, re.M)
    if not steps:
        return "No plan received."
    return "\n".join(f"Step {i}: completed - {s}" for i, s in enumerate(steps, 1))


def reviewer(user: str) -> str:
    """Approve when the executor output has at least 3 'Step N' lines, otherwise reject."""
    n = len(re.findall(r"^Step \d+", user, re.M))
    if n >= 3:
        return "APPROVED: all subtasks are covered."
    return f"REJECTED: only {n} of 3 subtasks completed. Planner must resend the full plan."


# role -> mock function; llm_clients.call looks agents up here (tests may swap entries)
MOCK = {"planner": planner, "executor": executor, "reviewer": reviewer}
