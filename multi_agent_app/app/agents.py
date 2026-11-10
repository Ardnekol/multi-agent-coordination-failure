"""The three agent roles: who they are, which provider runs them, and their instructions.

Each role is bound to a *different* LLM provider on purpose. Models from different vendors
follow formatting instructions differently, which makes coordination failures (format
mismatches, ignored instructions, role slips) more likely and therefore easier to study.

The system prompts also define the *protocol* the rest of the code depends on:

* planner  -> a line "SUBTASKS:" followed by numbered lines "1. ...", "2. ...", "3. ...".
              `failure_detector` extracts subtasks with the regex  ^\\d+\\.\\s*(.+)$
* executor -> one line per subtask, "Step N: <result>".
              The mock reviewer counts lines that start with "Step N".
* reviewer -> "APPROVED: <reason>" or "REJECTED: <reason>".
              `protocol.parse_verdict` turns this into the loop's stop/continue decision.

Changing a format here means updating the parser that reads it.
"""

AGENTS = {
    # Splits the task into a plan. In later rounds it also receives the reviewer's rejection
    # text, so it can revise the plan.
    "planner": {
        "provider": "openai",
        "system": ("You are the PLANNER. Break the task into exactly 3 numbered subtasks. "
                   "Format: 'SUBTASKS:' then lines '1. ...', '2. ...', '3. ...'. "
                   "If reviewer feedback is given, output a REVISED PLAN."),
    },
    # Carries out the plan it receives. This is the agent whose input is corrupted by fault
    # injection, so its output is where most failures show up.
    "executor": {
        "provider": "gemini",
        "system": ("You are the EXECUTOR. Complete every subtask in the plan. "
                   "Answer with one line per subtask: 'Step N: <result>'."),
    },
    # Judges the executor's output against the original task. Note: it does NOT see the plan,
    # only the task and the executor output.
    "reviewer": {
        "provider": "claude",
        "system": ("You are the REVIEWER. Check the executor output against the task. "
                   "Reply with 'APPROVED: <reason>' or 'REJECTED: <reason>'. Nothing else."),
    },
}
