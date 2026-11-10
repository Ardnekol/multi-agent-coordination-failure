"""Rule-based detector for multi-agent coordination failures, plus the coordination score.

`detect(task, msgs)` reads the finished conversation of one run and returns a list of failures.
It uses fixed rules (string length, keyword overlap, regexes, text similarity), not another
LLM, so results are fast, free, and reproducible - but heuristic. See README "Limitations".

Taxonomy (type -> severity 1..5, higher = worse):
  EMPTY_OR_TRUNCATED_HANDOFF  3   planner/executor message under 40 characters
  TASK_DRIFT                  4   under 25% of the task's keywords appear in the output
  INSTRUCTION_DROPPED         3   executor covered under 60% of the planned subtasks
  ROLE_CONFUSION              4   planner/executor replied with a reviewer verdict
  PROTOCOL_VIOLATION          2   reviewer reply contains no APPROVED/REJECTED verdict
  REPETITION_LOOP             2   executor output over 90% similar to its previous output
  REVIEW_PING_PONG            3   reviewer rejected 2 or more times
  NO_CONSENSUS_DEADLOCK       5   run ended without approval
  AGENT_CRASH                 3   provider API call raised an error

Each failure is a dict: {"step": int, "type": str, "severity": int, "detail": str}, where
`step` is the message the failure is attributed to.
"""
import re
import difflib
from .protocol import parse_verdict, is_strict_verdict

# Words ignored when comparing keywords. Besides common filler, this includes words that the
# agents' own output format adds to every message ("step", "subtasks", "completed", ...),
# which would otherwise make every output look related to every task.
STOP = {"this", "that", "with", "from", "have", "will", "about", "into", "their",
        "step", "subtasks", "completed", "plan", "revised"}

# Thresholds, named so they are easy to find and tune.
MIN_HANDOFF_CHARS = 40       # shorter planner/executor messages can't carry a real plan/result
DRIFT_MIN_OVERLAP = 0.25     # share of task keywords an output must contain
COVERAGE_MIN = 0.6           # share of planned subtasks the executor must address
REPEAT_SIMILARITY = 0.9      # difflib ratio above which two outputs count as "the same"


def kw(text):
    """Keyword set of a text: lowercase words of 4+ letters, minus STOP words.

    Example: kw("Explain how solar panels work") -> {"explain", "solar", "panels", "work"}.
    Only exact word forms match, so "panel" and "panels" are different keywords.
    """
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}


def _f(step, type_, sev, detail):
    """Build one failure record."""
    return {"step": step, "type": type_, "severity": sev, "detail": detail}


def detect(task, msgs):
    """Return every coordination failure found in one run.

    task  the original task string given to the agents
    msgs  the run's messages in order; each needs at least step, round, role and content
    """
    out, tk = [], kw(task)
    verdicts = [m for m in msgs if m["role"] == "reviewer"]

    # ---- 1. Per-message checks ---------------------------------------------------------------
    for m in msgs:
        c, r = m["content"].strip(), m["role"]

        # llm_clients.call turns an API exception into "[AGENT_ERROR] ..." text. The other
        # content checks would be meaningless on an error message, so skip them.
        if c.startswith("[AGENT_ERROR]"):
            out.append(_f(m["step"], "AGENT_CRASH", 3, c[:120]))
            continue

        if r in ("planner", "executor"):
            # Too short to be a usable plan or result (e.g. "[message lost]", "No plan received.")
            if len(c) < MIN_HANDOFF_CHARS:
                out.append(_f(m["step"], "EMPTY_OR_TRUNCATED_HANDOFF", 3,
                              f"{r} produced only {len(c)} characters"))
            # Long enough, but about something else. `elif`: an empty message isn't also drift.
            elif tk and len(kw(c) & tk) / len(tk) < DRIFT_MIN_OVERLAP:
                out.append(_f(m["step"], "TASK_DRIFT", 4, f"{r} output unrelated to the task"))
            # A non-reviewer that opens with a verdict has taken over the reviewer's job.
            if is_strict_verdict(c):
                out.append(_f(m["step"], "ROLE_CONFUSION", 4, f"{r} answered like the reviewer"))

        # markdown or a short preamble around the verdict is tolerated; a missing verdict is not
        if r == "reviewer" and parse_verdict(c) is None:
            out.append(_f(m["step"], "PROTOCOL_VIOLATION", 2, "reviewer gave no APPROVED/REJECTED verdict"))

    # ---- 2. Plan coverage: did the executor address the planner's subtasks? (per round) -----
    for rnd in sorted({m["round"] for m in msgs}):
        plan = next((m for m in msgs if m["round"] == rnd and m["role"] == "planner"), None)
        exe = next((m for m in msgs if m["round"] == rnd and m["role"] == "executor"), None)
        if plan and exe:
            # Note: this is the plan as the planner wrote it, before fault injection, so a
            # truncated or dropped handoff shows up here as missing subtasks.
            subs = re.findall(r"^\d+\.\s*(.+)$", plan["content"], re.M)
            if subs:
                # A subtask counts as done if the executor output shares a keyword with it.
                # Compare on subtask-specific words: task words appear in every subtask and
                # would make every subtask look done. Fall back to all words if none remain.
                done = sum(1 for s in subs if ((kw(s) - tk) or kw(s)) & kw(exe["content"]))
                # Very short outputs are already reported as EMPTY_OR_TRUNCATED_HANDOFF.
                if done / len(subs) < COVERAGE_MIN and len(exe["content"].strip()) >= MIN_HANDOFF_CHARS:
                    out.append(_f(exe["step"], "INSTRUCTION_DROPPED", 3,
                                  f"executor covered {done}/{len(subs)} planned subtasks"))

    # ---- 3. Repetition: the executor giving the same answer round after round ---------------
    for role in ("executor",):
        seq = [m for m in msgs if m["role"] == role]
        for a, b in zip(seq, seq[1:]):                     # consecutive pairs
            if difflib.SequenceMatcher(None, a["content"], b["content"]).ratio() > REPEAT_SIMILARITY:
                out.append(_f(b["step"], "REPETITION_LOOP", 2, f"{role} repeated previous output"))

    # ---- 4. Run-level outcome checks ---------------------------------------------------------
    rejects = [v for v in verdicts if parse_verdict(v["content"]) == "REJECTED"]
    if len(rejects) >= 2:
        out.append(_f(rejects[-1]["step"], "REVIEW_PING_PONG", 3, f"{len(rejects)} rejections"))
    # The orchestrator stops at the first approval, so only the last verdict decides the outcome.
    if not verdicts or parse_verdict(verdicts[-1]["content"]) != "APPROVED":
        out.append(_f(msgs[-1]["step"] if msgs else 0, "NO_CONSENSUS_DEADLOCK", 5,
                      "run finished without reviewer approval"))
    return out


def score(failures):
    """Coordination score from 0 to 100: start at 100, subtract 5 points per severity point.

    Example: TASK_DRIFT (4) + INSTRUCTION_DROPPED (3) -> 100 - 5 * 7 = 65.
    The result never goes below 0. The weights are a design choice, not a measured quantity.
    """
    return max(0, 100 - 5 * sum(f["severity"] for f in failures))
