"""Share run history through git: export runs to JSON, and load them into an empty database.

The SQLite database (coordination.db) is git-ignored: it is binary and changes on every run,
so committing it would cause merge conflicts. Instead, runs worth sharing are exported to a
readable JSON file that is committed (data/sample_runs.json by default). When the API starts
with an empty database, it loads that file, so anyone who clones or pulls the repo sees the
same history in the dashboard. Runs keep their original ids.

Command line (run from the project folder):
    python -m app.seed export                  # write all stored runs to data/sample_runs.json
    python -m app.seed export my_runs.json     # ...or to another file
    python -m app.seed import                  # load the file (only into an empty database)
"""
import json
import os
import sys
from datetime import datetime, timezone

from . import database as db

DEFAULT_PATH = os.getenv("SEED_PATH", os.path.join("data", "sample_runs.json"))

# Only these columns are written back, so a hand-edited file can't inject other SQL.
_COLUMNS = {
    "runs": ["id", "task", "mode", "fault", "status", "score", "created_at", "fault_outcome",
             "fault_rounds", "max_rounds"],
    "messages": ["run_id", "step", "round", "agent", "role", "content", "latency_ms", "source"],
    "failures": ["run_id", "step", "type", "severity", "detail"],
}


def export_runs(path=DEFAULT_PATH):
    """Write every finished run (with its messages and failures) to `path`. Returns the count."""
    db.init_db()
    with db._conn() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM runs WHERE status != 'running' ORDER BY id")]
    runs = [db.get_run(i) for i in ids]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"format": 1, "exported_at": datetime.now(timezone.utc).isoformat(), "runs": runs},
                  f, ensure_ascii=False, indent=2)
        f.write("\n")
    return len(runs)


def import_runs(path=DEFAULT_PATH):
    """Load runs from `path` into the database, but only if it has no runs yet.

    Returns the number of runs imported (0 when the file is missing or the database already has
    runs, so it is safe to call on every startup and never duplicates or overwrites anything).
    """
    if not os.path.exists(path):
        return 0
    db.init_db()
    with db._conn() as c:
        if c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]:
            return 0
        with open(path, encoding="utf-8") as f:
            runs = json.load(f).get("runs", [])
        for record in runs:
            _insert(c, "runs", record["run"])
            for m in record.get("messages", []):
                _insert(c, "messages", m)
            for fl in record.get("failures", []):
                _insert(c, "failures", fl)
    return len(runs)


def _insert(conn, table, row):
    cols = [k for k in _COLUMNS[table] if k in row]
    conn.execute(f"INSERT INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                 [row[k] for k in cols])


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    target = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PATH
    if action == "export":
        print(f"exported {export_runs(target)} runs to {target}")
    elif action == "import":
        n = import_runs(target)
        print(f"imported {n} runs from {target}" if n else "nothing imported (file missing or database not empty)")
    else:
        sys.exit(__doc__)
