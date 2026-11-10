"""FastAPI backend: the HTTP interface to the lab.

Start it with `uvicorn app.main:app --reload` from the project folder; interactive API
docs are then at http://127.0.0.1:8000/docs. The Streamlit UI is just a client of these
endpoints, so anything the UI does can also be scripted with plain HTTP requests.

Endpoints:
  POST /runs        run one experiment and return its full record
  GET  /runs        the 100 most recent runs (summary rows)
  GET  /runs/{id}   one run with its messages and failures
  GET  /stats       aggregates for the charts
  GET  /faults      the fault names accepted by POST /runs
  GET  /health      liveness check
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from . import database as db
from .orchestrator import run_task, MissingKeysError
from .fault_injection import FAULTS

app = FastAPI(title="Multi-Agent Coordination Failure Lab", version="1.0")
# Allow any origin so a browser-based frontend on another port can call the API.
# Fine for a local research tool; restrict this before exposing the server publicly.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
db.init_db()                                  # create/upgrade tables at startup


class RunRequest(BaseModel):
    """Body of POST /runs. Pydantic rejects out-of-range values with a 422 response."""
    task: str = Field(min_length=5, max_length=500)    # what the agents should do
    live: bool = False                       # False = mock agents (no API keys needed)
    fault: str = "none"                      # one of FAULTS (checked in create_run)
    fault_rounds: int = Field(1, ge=0, le=3)  # inject the fault in rounds 1..fault_rounds
    max_rounds: int = Field(3, ge=1, le=5)    # stop (deadlock) after this many rounds without approval
    allow_mock_fallback: bool = False        # live only: run keyless providers as mocks ('mixed' run)


@app.get("/health")
def health():
    """Liveness check for scripts and monitors."""
    return {"status": "ok"}


@app.get("/faults")
def faults():
    """Fault names accepted by POST /runs (used to fill the UI dropdown)."""
    return FAULTS


@app.post("/runs")
def create_run(req: RunRequest):
    """Run one experiment synchronously and return {"run", "messages", "failures"}.

    Responds 400 for an unknown fault, or for live mode with missing API keys (unless
    allow_mock_fallback is true). In live mode this waits for up to 3 x max_rounds real
    API calls, so it can take a while.
    """
    if req.fault not in FAULTS:
        raise HTTPException(400, f"fault must be one of {FAULTS}")
    try:
        return run_task(req.task, req.live, req.fault, req.fault_rounds, req.max_rounds,
                        req.allow_mock_fallback)
    except MissingKeysError as e:
        raise HTTPException(400, str(e))


@app.get("/runs")
def runs():
    """The 100 most recent runs, newest first."""
    return db.list_runs()


@app.get("/runs/{run_id}")
def run(run_id: int):
    """Full record of one run; 404 if the id does not exist."""
    r = db.get_run(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    return r


@app.get("/stats")
def stats():
    """Failure counts by type and average score by fault, across all stored runs."""
    return db.stats()
