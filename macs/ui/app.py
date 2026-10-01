"""MACS dashboard — run sessions, browse traces, compare conditions, in the browser.

Arush - this is the visual front end for your project. It does three things, one per tab:
  * Results  - scores every trace and shows the A/B/C/D/D_minus comparison (families, novelty,
               correctness) as a table and a bar chart. This is your headline result, live.
  * Traces   - pick any session and read it as a tidy timeline: who said what, which strategies the
               studio proposed, and whether each passed - the evidence behind the numbers.
  * Run      - start a new session (pick a task, a condition, a seed) without touching the terminal.
               Needs Ollama running, same as the CLI.

Run it with:   streamlit run macs/ui/app.py

The data-loading logic is kept in plain functions (top of the file) so it can be tested without
Streamlit; only the layout below imports `st`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "data" / "traces"
RESULTS_FILE = ROOT / "data" / "results" / "summary.json"

CONDITIONS = ["A", "B", "C", "D", "D_minus"]
COND_DESC = {
    "A": "1 learner + Tutor (baseline)",
    "B": "1 learner + Studio",
    "C": "2 learners + Tutor",
    "D": "2 learners + Studio",
    "D_minus": "2 learners + Studio, Facilitator OFF",
}
# tutor conditions vs studio conditions - used to colour the chart
TUTOR = {"A", "C"}

# ----------------------------------------------------------------- pure helpers (no Streamlit)

def list_traces() -> list[Path]:
    return sorted(TRACES_DIR.glob("*.jsonl"))


def read_events(path: Path) -> list[dict]:
    from macs.trace.writer import read_trace
    return list(read_trace(path))


def trace_header(events: list[dict]) -> dict:
    start = next((e for e in events if e.get("event") == "session_start"), {})
    return {
        "task_id": start.get("task_id", "?"),
        "condition": start.get("condition", "?"),
        "session_id": start.get("session_id", "?"),
        "n_events": len(events),
    }


def event_rows(events: list[dict]) -> list[dict]:
    """One tidy row per event for the timeline table."""
    rows = []
    for e in events:
        content = (e.get("content") or "").replace("\n", " ")
        rows.append({
            "seq": e.get("seq"),
            "role": e.get("role", ""),          # who acted: learner_1, innovator, engineer, critic, verifier…
            "event": e.get("event", ""),
            "detail": content[:120],
        })
    return rows


def strategies(events: list[dict]) -> list[str]:
    """Pull the studio's proposed strategy names out of the proposal events."""
    out = []
    for e in events:
        if e.get("event") == "proposal":
            text = e.get("content", "")
            name = None
            for line in text.splitlines():
                if "Strategy" in line:
                    name = line.split("Strategy", 1)[1].lstrip("*: ").strip()
                    break
            out.append(name or (text[:40] + "..."))
    return out


def score_all() -> tuple[list[dict], dict]:
    """Score every trace and aggregate by condition (loads reference sets for novelty)."""
    from macs.analysis.aggregate import analyse_dir
    references = {}
    try:
        from macs.metrics.reference import build_reference_stats, REFERENCE_ROOT
        for f in Path(REFERENCE_ROOT).glob("*.jsonl"):
            try:
                references[f.stem] = build_reference_stats(f.stem)
            except Exception:  # noqa: BLE001 - an empty/old reference file is just skipped
                pass
    except Exception:  # noqa: BLE001
        pass
    return analyse_dir(str(TRACES_DIR), backend="subprocess", references=references)


def run_session(task_dir: str, condition: str, seed: int) -> subprocess.CompletedProcess:
    """Start one session via the same CLI the terminal uses (needs Ollama running)."""
    cmd = [sys.executable, str(ROOT / "scripts" / "run_session.py"),
           "--task", task_dir, "--condition", condition, "--seed", str(seed), "--sandbox", "subprocess"]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)


# ----------------------------------------------------------------- Streamlit UI

def main() -> None:  # pragma: no cover - the layout itself isn't unit-tested
    import pandas as pd
    import streamlit as st
    from macs.tasks.bank import load_bank

    st.set_page_config(page_title="MACS dashboard", layout="wide")
    st.title("Multi-Agent Coding Studio — dashboard")
    st.caption("Trace-based creativity measurement · run sessions, read traces, compare conditions")

    tab_results, tab_traces, tab_run = st.tabs(["📊 Results", "🔍 Traces", "▶ Run a session"])

    # ---- Results ----
    with tab_results:
        st.subheader("Condition comparison")
        st.write("Families = how many genuinely different solution approaches each setup produced; "
                 "novelty = how far they sit from the ordinary solution.")
        if st.button("Re-score all traces", type="primary"):
            with st.spinner("scoring traces…"):
                st.session_state["scored"] = score_all()
        scored = st.session_state.get("scored")
        if scored is None and RESULTS_FILE.exists():
            data = json.loads(RESULTS_FILE.read_text())
            scored = (data.get("sessions", []), data.get("summary", {}))
        if not scored:
            st.info("No results yet. Click **Re-score all traces** (or run a session first).")
        else:
            _sessions, summary = scored
            rows = []
            for cond in CONDITIONS:
                m = summary.get(cond)
                if not m:
                    continue
                rows.append({
                    "Condition": cond, "Setup": COND_DESC[cond], "Sessions": m["n_sessions"],
                    "Families": m.get("avg_families"), "Novelty (z)": m.get("avg_mean_novelty"),
                    "Correct": m.get("avg_correct"), "Heuristic": m.get("avg_heuristic_score"),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            fam = df.dropna(subset=["Families"]).set_index("Condition")["Families"]
            if not fam.empty:
                st.bar_chart(fam, height=300)   # families by condition: tutors ~1, studios ~6
                st.caption("Solution families by condition — the project's core creativity contrast.")

    # ---- Traces ----
    with tab_traces:
        st.subheader("Browse a session trace")
        traces = list_traces()
        if not traces:
            st.info("No traces in data/traces/ yet.")
        else:
            pick = st.selectbox("Session", traces, format_func=lambda p: p.name)
            events = read_events(pick)
            h = trace_header(events)
            c1, c2, c3 = st.columns(3)
            c1.metric("Task", h["task_id"])
            c2.metric("Condition", h["condition"])
            c3.metric("Events", h["n_events"])
            strat = strategies(events)
            if strat:
                st.write("**Strategies the studio proposed:**")
                st.write(" · ".join(f"`{s}`" for s in strat))
            st.write("**Timeline**")
            st.dataframe(pd.DataFrame(event_rows(events)), hide_index=True, use_container_width=True)

    # ---- Run ----
    with tab_run:
        st.subheader("Run a new session")
        st.write("Starts one session with the local model. **Ollama must be running.**")
        tasks = {t.id: t for t in load_bank()}
        task_id = st.selectbox("Task", sorted(tasks), index=0)
        condition = st.selectbox("Condition", CONDITIONS, format_func=lambda c: f"{c} — {COND_DESC[c]}")
        seed = st.number_input("Seed", min_value=1, value=1, step=1)
        if st.button("Run session", type="primary"):
            task = tasks[task_id]
            task_dir = str(task.path)
            with st.spinner(f"running {task_id} / {condition}… (this calls the model)"):
                try:
                    res = run_session(task_dir, condition, int(seed))
                    st.code(res.stdout[-4000:] or "(no output)")
                    if res.returncode != 0:
                        st.error("Session failed:")
                        st.code(res.stderr[-2000:])
                    else:
                        st.success("Done. Open the Results tab and re-score to include it.")
                except subprocess.TimeoutExpired:
                    st.error("Timed out after 30 min.")


if __name__ == "__main__":
    main()
