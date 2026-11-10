"""Streamlit dashboard for the lab: run experiments and look at the results.

Run with `streamlit run frontend/streamlit_app.py` while the FastAPI backend is running on
port 8000 (`uvicorn app.main:app`). This page holds no logic of its own; it only calls the
REST API, so it shows exactly what the API returns.

Layout:
  sidebar   experiment settings and the Run button
  main      result of the last run: score, outcome, mode, fault outcome, conversation, failures
  bottom    aggregate charts over every stored run
"""
import requests
import pandas as pd
import streamlit as st

API = "http://127.0.0.1:8000"          # change if the backend runs elsewhere

st.set_page_config(page_title="Multi-Agent Failure Lab", layout="wide")
st.title("Multi-Agent Coordination Failure Lab")
st.caption("Planner (OpenAI) -> Executor (Gemini) -> Reviewer (Claude)")

# ---- Experiment settings ------------------------------------------------------------------
# Streamlit re-runs this whole script on every interaction; `go` is True only on the run
# in which the button was clicked.
with st.sidebar:
    st.header("Experiment")
    task = st.text_area("Task", "Explain how solar panels generate electricity")
    fault = st.selectbox("Inject fault", requests.get(f"{API}/faults").json())
    live = st.checkbox("Use live LLM APIs (needs keys in .env)", value=False)
    fallback = st.checkbox("Live: use mocks for providers without a key", value=False, disabled=not live)
    fault_rounds = st.slider("Fault active for N rounds", 0, 3, 1)
    max_rounds = st.slider("Max rounds", 1, 5, 3)
    go = st.button("Run experiment", type="primary")

# ---- Result of one run ----------------------------------------------------------------------
if go:
    with st.spinner("Agents are coordinating..."):
        r = requests.post(f"{API}/runs", json={"task": task, "live": live, "fault": fault,
                                               "fault_rounds": fault_rounds, "max_rounds": max_rounds,
                                               "allow_mock_fallback": fallback})
    if r.ok:
        d = r.json()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Coordination score", d["run"]["score"])
        c2.metric("Outcome", d["run"]["status"])                 # approved / deadlock
        c3.metric("Failures detected", len(d["failures"]))
        c4.metric("Mode", d["run"]["mode"])                      # mock / live / mixed
        c5.metric("Fault", d["run"].get("fault_outcome") or "n/a")   # effective / resisted / n/a
        if d["run"]["mode"] == "mixed":
            st.warning("Some agents ran as mocks because their API key is missing. "
                       "Don't compare this run with fully live runs.")

        # Every message in order. Note: the planner's text is shown as it wrote it; the
        # executor may have received a corrupted copy (fault injection).
        st.subheader("Conversation log")
        for m in d["messages"]:
            with st.chat_message("assistant"):
                st.markdown(f"**R{m['round']} - {m['role']} ({m['agent']})** - {m['latency_ms']} ms")
                st.text(m["content"])

        st.subheader("Detected failures")
        if d["failures"]:
            st.dataframe(pd.DataFrame(d["failures"]), use_container_width=True)
        else:
            st.success("No coordination failures detected")
    else:
        st.error(r.text)                 # e.g. 400 "live mode needs API keys for: ..."

# ---- Aggregates over all stored runs --------------------------------------------------------
st.divider()
st.subheader("Aggregate analysis")
s = requests.get(f"{API}/stats").json()
a, b = st.columns(2)
if s["failures_by_type"]:
    a.write("Failures by type")
    a.bar_chart(pd.DataFrame(s["failures_by_type"]).set_index("type"))
if s["score_by_fault"]:
    b.write("Average coordination score by injected fault")
    b.bar_chart(pd.DataFrame(s["score_by_fault"]).set_index("fault")["avg_score"])
