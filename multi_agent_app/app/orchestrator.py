"""Runs one experiment: the Planner -> Executor -> Reviewer loop, with logging and detection.

One *round* is three agent calls:

    planner  (OpenAI)  task (+ last rejection)  -> plan
       |  fault injection happens here, in rounds 1..fault_rounds only
       v
    executor (Gemini)  task + (possibly corrupted) plan -> result
    reviewer (Claude)  task + result                     -> APPROVED / REJECTED

The loop stops at the first APPROVED verdict, or after `max_rounds` rounds (a deadlock).
After the loop, the whole conversation is analysed by `failure_detector.detect`, scored, and
the run record is completed in SQLite.
"""
from . import database as db
from .agents import AGENTS
from .llm_clients import call, missing_keys
from .fault_injection import inject
from .failure_detector import detect, score
from .protocol import parse_verdict


class MissingKeysError(ValueError):
    """Live mode was requested but some providers have no API key."""


def _run_mode(sources):
    """'live' only if every message came from a real API call, 'mock' if none did.

    `sources` are the per-message values from llm_clients.call ('live', 'error', 'mock').
    'error' counts as live: the real API was called, it just failed.
    Anything in between is 'mixed', which should not be compared with fully live runs.
    """
    real = {"live", "error"}
    if all(s in real for s in sources):
        return "live"
    if not any(s in real for s in sources):
        return "mock"
    return "mixed"


def fault_outcome(fault, fault_rounds, msgs, failures):
    """Did the injected fault actually disrupt the run?

    'effective' - a failure was detected in an injected round, or that round's reviewer did not
                  approve; 'resisted' - the agents worked through the corrupted handoff cleanly;
    'n/a'       - no fault was injected.
    Real models often ignore injected instructions, so this separates "fault had no effect"
    from "detector missed it" when you compare runs.
    """
    if fault == "none" or fault_rounds == 0 or not msgs:
        return "n/a"
    # rounds that really ran AND had the fault (a run approved in round 1 never reaches round 2)
    injected = {m["round"] for m in msgs if m["round"] <= fault_rounds}
    if not injected:
        return "n/a"
    round_of = {m["step"]: m["round"] for m in msgs}      # failures carry a step, not a round
    if any(round_of.get(f["step"]) in injected for f in failures):
        return "effective"
    for m in msgs:
        if m["role"] == "reviewer" and m["round"] in injected and parse_verdict(m["content"]) != "APPROVED":
            return "effective"
    return "resisted"


def run_task(task, live=False, fault="none", fault_rounds=1, max_rounds=3, allow_mock_fallback=False):
    """Run one experiment and return the stored record {"run", "messages", "failures"}.

    task                 what the agents should do
    live                 False: mock agents; True: real provider APIs
    fault                one of fault_injection.FAULTS
    fault_rounds         the fault is applied in rounds 1..fault_rounds (0 = never)
    max_rounds           stop with status 'deadlock' after this many rounds without approval
    allow_mock_fallback  live only: let providers without a key run as mocks (run is 'mixed')

    Raises MissingKeysError for live mode with missing keys and no fallback. The check runs
    before anything is written, so a refused request leaves no half-finished run behind.
    """
    if live and not allow_mock_fallback:
        missing = missing_keys({a["provider"] for a in AGENTS.values()})
        if missing:
            raise MissingKeysError(
                f"live mode needs API keys for: {', '.join(missing)}. Add them to .env, "
                "or set allow_mock_fallback to run those agents as mocks (the run is then marked 'mixed').")

    run_id = db.create_run(task, "live" if live else "mock", fault)
    msgs, step, feedback, approved = [], 0, "", False

    def ask(rnd, role, user):
        """Call one agent, log the message (in memory and in the DB), return its text."""
        nonlocal step
        a = AGENTS[role]
        text, lat, source = call(a["provider"], role, a["system"], user, live)
        step += 1
        m = dict(step=step, round=rnd, agent=a["provider"], role=role, content=text,
                 latency_ms=lat, source=source)
        msgs.append(m)
        db.add_message(run_id, m)
        return text

    for rnd in range(1, max_rounds + 1):
        # From round 2 on, the planner sees the reviewer's last rejection so it can revise.
        p_user = f"Task: {task}\n" + (f"Reviewer feedback to address: {feedback}" if feedback else "")
        plan = ask(rnd, "planner", p_user)

        # The log keeps the planner's original plan; only the executor sees the corrupted copy.
        handoff = inject(plan, fault) if rnd <= fault_rounds else plan   # fault point
        result = ask(rnd, "executor", f"Task: {task}\nPlan:\n{handoff}")
        verdict = ask(rnd, "reviewer", f"Task: {task}\nExecutor output:\n{result}")

        if parse_verdict(verdict) == "APPROVED":
            approved = True
            break
        feedback = verdict                    # anything but APPROVED (incl. no verdict) loops

    # Analyse the finished conversation, then complete the run record.
    failures = detect(task, msgs)
    sc = score(failures)
    db.add_failures(run_id, failures)
    db.finish_run(run_id, "approved" if approved else "deadlock", sc,
                  mode=_run_mode([m["source"] for m in msgs]),
                  fault_outcome=fault_outcome(fault, fault_rounds, msgs, failures))
    return db.get_run(run_id)
