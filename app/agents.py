"""The three agent roles: who they are, which model runs them, and their instructions.

The provider and model of each role come from config.ROLE_MODELS. By default all three run on
Claude with a different model size each (Opus plans, Sonnet executes, Haiku reviews). Agents
of different capability are a realistic team and make coordination failures more likely: a
small reviewer may judge differently from a large planner, and so on. Set <ROLE>_PROVIDER in
.env to mix vendors instead (e.g. the original OpenAI planner / Gemini executor setup).

The system prompts also define the *protocol* the rest of the code depends on:

* planner  -> a line "SUBTASKS:" followed by numbered lines "1. ...", "2. ...", "3. ...".
              `failure_detector` extracts subtasks with the regex  ^\\d+\\.\\s*(.+)$
* executor -> one line per subtask, "Step N: <result>".
              The mock reviewer counts lines that start with "Step N".
* reviewer -> "APPROVED: <reason>" or "REJECTED: <reason>".
              `protocol.parse_verdict` turns this into the loop's stop/continue decision.

Changing a format here means updating the parser that reads it.
"""
from .config import ROLE_MODELS

_PROMPTS = {
    # Splits the task into a plan. In later rounds it also receives the reviewer's rejection
    # text, so it can revise the plan.
    "planner": ("You are the PLANNER. Break the task into exactly 3 numbered subtasks. "
                "Format: 'SUBTASKS:' then lines '1. ...', '2. ...', '3. ...'. "
                "If reviewer feedback is given, output a REVISED PLAN."),
    # Carries out the plan it receives. This is the agent whose input is corrupted by fault
    # injection, so its output is where most failures show up.
    "executor": ("You are the EXECUTOR. Complete every subtask in the plan. "
                 "Answer with one line per subtask: 'Step N: <result>'."),
    # Judges the executor's output against the original task. Note: it does NOT see the plan,
    # only the task and the executor output.
    "reviewer": ("You are the REVIEWER. Check the executor output against the task. "
                 "Reply with 'APPROVED: <reason>' or 'REJECTED: <reason>'. Nothing else."),
}

# role -> {"provider": "claude" | "openai" | "gemini", "model": str, "system": str}
AGENTS = {
    role: {"provider": ROLE_MODELS[role][0], "model": ROLE_MODELS[role][1], "system": prompt}
    for role, prompt in _PROMPTS.items()
}
