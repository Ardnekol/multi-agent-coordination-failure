"""Export / import of shared run history (app/seed.py)."""
import json
from app import config, seed
from app.orchestrator import run_task


def test_export_then_import_into_empty_database(tmp_path, monkeypatch):
    run_task("Explain how the attention mechanism works", fault="contradict")
    path = tmp_path / "runs.json"
    exported = seed.export_runs(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert exported >= 1 and len(data["runs"]) == exported

    # a fresh, empty database receives every run with the same ids, messages and failures
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "fresh.db"))
    assert seed.import_runs(str(path)) == exported
    from app import database as db
    original = data["runs"][-1]
    copy = db.get_run(original["run"]["id"])
    assert copy["run"]["score"] == original["run"]["score"]
    assert [m["content"] for m in copy["messages"]] == [m["content"] for m in original["messages"]]
    assert {f["type"] for f in copy["failures"]} == {f["type"] for f in original["failures"]}

    # never imports twice into a database that already has runs
    assert seed.import_runs(str(path)) == 0


def test_missing_file_imports_nothing(tmp_path):
    assert seed.import_runs(str(tmp_path / "does-not-exist.json")) == 0
