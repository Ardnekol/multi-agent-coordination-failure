"""SQLite persistence for experiments.

Schema (one run has many messages and many failures, linked by run_id):

    runs      id, task, mode, fault, status, score, created_at, fault_outcome, fault_rounds, max_rounds
              mode          'mock' | 'live' | 'mixed'  (see orchestrator._run_mode)
              status        'running' while in progress, then 'approved' | 'deadlock'
              fault_outcome 'n/a' | 'effective' | 'resisted' (see orchestrator.fault_outcome)

    messages  id, run_id, step, round, agent, role, content, latency_ms, source
              step    1, 2, 3, ... across the whole run (planner, executor, reviewer, planner, ...)
              round   1..max_rounds; one round = planner + executor + reviewer
              agent   provider name ('openai', 'gemini', 'claude'); role = 'planner', ...
              source  'live' | 'mock' | 'error' - where this answer actually came from

    failures  id, run_id, step, type, severity, detail
              step points at the message where the failure was detected

Every function opens its own short-lived connection. That is simple and safe for a small
single-user app; `with conn:` commits on success and rolls back on an exception.
"""
import sqlite3
from datetime import datetime, timezone
from . import config


def _conn():
    """Open a connection whose rows can be read like dicts (row["column"])."""
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Create the tables if they don't exist, and add columns introduced in later versions.

    Safe to call any number of times. `CREATE TABLE IF NOT EXISTS` never changes an existing
    table, so newer columns are added with ALTER TABLE; SQLite raises OperationalError when the
    column already exists, which is the "nothing to do" case.
    """
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT, mode TEXT, fault TEXT,
            status TEXT DEFAULT 'running', score INTEGER,
            created_at TEXT);
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER, step INTEGER, round INTEGER,
            agent TEXT, role TEXT, content TEXT, latency_ms INTEGER);
        CREATE TABLE IF NOT EXISTS failures(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER, step INTEGER, type TEXT,
            severity INTEGER, detail TEXT);
        """)
        # columns added after the first release; ALTER keeps existing databases usable
        for table, column in (("runs", "fault_outcome TEXT"), ("messages", "source TEXT"),
                              ("runs", "fault_rounds INTEGER"), ("runs", "max_rounds INTEGER")):
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
            except sqlite3.OperationalError:
                pass                                # column already exists


def create_run(task, mode, fault, fault_rounds=None, max_rounds=None):
    """Insert a new run in status 'running' and return its id.

    `mode` here is the *requested* mode; `finish_run` overwrites it with the mode that actually
    happened (e.g. 'mixed' if some agents fell back to mocks). fault_rounds / max_rounds are
    stored so the dashboard can show where the fault was active in past runs.
    """
    init_db()
    with _conn() as c:
        cur = c.execute("INSERT INTO runs(task,mode,fault,fault_rounds,max_rounds,created_at) VALUES(?,?,?,?,?,?)",
                        (task, mode, fault, fault_rounds, max_rounds, datetime.now(timezone.utc).isoformat()))
        return cur.lastrowid


def add_message(run_id, m):
    """Store one agent message. Called immediately, so a crashed run still leaves its log."""
    with _conn() as c:
        c.execute("INSERT INTO messages(run_id,step,round,agent,role,content,latency_ms,source) "
                  "VALUES(?,?,?,?,?,?,?,?)",
                  (run_id, m["step"], m["round"], m["agent"], m["role"], m["content"], m["latency_ms"],
                   m.get("source")))


def add_failures(run_id, failures):
    """Store the list of failure dicts produced by `failure_detector.detect`."""
    with _conn() as c:
        c.executemany("INSERT INTO failures(run_id,step,type,severity,detail) VALUES(?,?,?,?,?)",
                      [(run_id, f["step"], f["type"], f["severity"], f["detail"]) for f in failures])


def finish_run(run_id, status, score, mode=None, fault_outcome=None):
    """Record the final status, score, actual mode and fault outcome of a run.

    COALESCE keeps the mode stored at creation when `mode` is None.
    """
    with _conn() as c:
        c.execute("UPDATE runs SET status=?, score=?, mode=COALESCE(?, mode), fault_outcome=? WHERE id=?",
                  (status, score, mode, fault_outcome, run_id))


def list_runs():
    """The 100 most recent runs, newest first (summary rows only, no messages)."""
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 100")]


def get_run(run_id):
    """Full record of one run: {"run": {...}, "messages": [...], "failures": [...]}, or None."""
    with _conn() as c:
        run = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            return None
        msgs = [dict(r) for r in c.execute("SELECT * FROM messages WHERE run_id=? ORDER BY step", (run_id,))]
        fails = [dict(r) for r in c.execute("SELECT * FROM failures WHERE run_id=? ORDER BY step", (run_id,))]
        return {"run": dict(run), "messages": msgs, "failures": fails}


def stats():
    """Aggregates for the dashboard charts.

    failures_by_type  how often each failure type was detected, most common first
    score_by_fault    number of finished runs and average score per injected fault
    Note: these mix mock, live and mixed runs; filter by `mode` before drawing conclusions.
    """
    with _conn() as c:
        by_type = [dict(r) for r in c.execute(
            "SELECT type, COUNT(*) AS count FROM failures GROUP BY type ORDER BY count DESC")]
        by_fault = [dict(r) for r in c.execute(
            "SELECT fault, COUNT(*) AS runs, ROUND(AVG(score),1) AS avg_score "
            "FROM runs WHERE score IS NOT NULL GROUP BY fault")]
        return {"failures_by_type": by_type, "score_by_fault": by_fault}
