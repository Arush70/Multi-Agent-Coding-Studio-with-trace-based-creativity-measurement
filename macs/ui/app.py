"""MACS dashboard - run sessions, read traces, and SEE the creativity measurement.

Arush - this is the visual front end for your project, rebuilt to actually explain itself. Tabs:
  * Overview   - the headline Studio-vs-Tutor comparison, as metric cards + charts, with a plain
                 'what this means' for every number.
  * Creativity - how creativity is measured and how to read it: diversity, novelty, correctness, and
                 the joint 'correct & novel' score, each defined with direction and a worked note.
  * Traces     - open any session: the strategies it proposed, each proposal's CODE, whether it
                 passed, how novel it is, and any Facilitator interventions.
  * Run        - start sessions without the terminal: one condition, or all five at once; pick the
                 task and sandbox backend. Needs Ollama running.

Run it with:   streamlit run macs/ui/app.py

Data-loading logic is kept in plain functions (top of file) so it can be tested without Streamlit;
only the layout imports `st`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "data" / "traces"
RESULTS_FILE = ROOT / "data" / "results" / "summary.json"
NOVELTY_PERCENTILE = 0.90

CONDITIONS = ["A", "B", "C", "D", "D_minus"]
COND_DESC = {
    "A": "1 learner + Tutor (baseline)",
    "B": "1 learner + Studio",
    "C": "2 learners + Tutor",
    "D": "2 learners + Studio",
    "D_minus": "2 learners + Studio, Facilitator OFF",
}
TUTOR = {"A", "C"}
STUDIO = {"B", "D", "D_minus"}

# plain-English meaning of each metric: (short label, what it is, which way is better)
METRICS = {
    "families": ("Diversity (families)",
                 "How many genuinely DIFFERENT solution approaches the system produced. A single tutor "
                 "usually gives 1; the studio explores several. This is set-level creativity.", "higher"),
    "novelty": ("Novelty (z)",
                "How far the solutions sit from the ORDINARY solution, in standard deviations, measured "
                "against a frozen reference set of normal solutions. 0 = typical, 2 = very unusual.", "higher"),
    "correct": ("Correctness",
                "Solutions that pass the hidden held-out tests (Tier 1/2) or are feasible (Tier 3). "
                "Creativity only counts if the code actually works.", "higher"),
    "creativity": ("Creativity score (correct & novel)",
                   "The headline: the share of proposals that are BOTH novel AND correct - new ideas that "
                   "actually work, not novelty for its own sake.", "higher"),
}

# ----------------------------------------------------------------- pure helpers (no Streamlit)

def list_traces() -> list[Path]:
    return sorted(TRACES_DIR.glob("*.jsonl"))


def read_events(path: Path) -> list[dict]:
    from macs.trace.writer import read_trace
    return list(read_trace(path))


def trace_header(events: list[dict]) -> dict:
    start = next((e for e in events if e.get("event") == "session_start"), {})
    return {"task_id": start.get("task_id", "?"), "condition": start.get("condition", "?"),
            "session_id": start.get("session_id", "?"), "n_events": len(events)}


def _parse_strategy(text: str) -> str:
    for line in (text or "").splitlines():
        if "Strategy" in line:
            return line.split("Strategy", 1)[1].lstrip("*: ").strip()
    return ""


def strategies(events: list[dict]) -> list[str]:
    return [_parse_strategy(e.get("content", "")) or "(unnamed)"
            for e in events if e.get("event") == "proposal"]


def interventions(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("event") == "intervention"]


def event_rows(events: list[dict]) -> list[dict]:
    rows = []
    for e in events:
        rows.append({"seq": e.get("seq"), "role": e.get("role", ""), "event": e.get("event", ""),
                     "detail": (e.get("content") or "").replace("\n", " ")[:120]})
    return rows


def _reference_for(task_id: str):
    try:
        from macs.metrics.reference import build_reference_stats
        return build_reference_stats(task_id)
    except Exception:  # noqa: BLE001 - no/empty reference -> novelty simply unavailable
        return None


def proposal_table(path: Path, backend: str = "subprocess") -> list[dict]:
    """Per-proposal breakdown: strategy, size, correctness, and novelty. The evidence behind a session."""
    import numpy as np
    from macs.analysis.score import implementations, task_by_id, score_code_heldout
    from macs.metrics.represent import ast_features, normalise
    events = read_events(path)
    h = trace_header(events)
    task = task_by_id(h["task_id"])
    ref = _reference_for(h["task_id"])
    strat_by_id = {e.get("proposal_id", ""): _parse_strategy(e.get("content", ""))
                   for e in events if e.get("event") == "proposal"}
    rows = []
    for pid, code in implementations(events):
        row = {"proposal": pid,
               "strategy": strat_by_id.get(pid) or ("revision" if "rev" in pid else "-"),
               "lines": code.count("\n") + 1, "code": code}
        if task.scoring == "tests":
            row["correct"] = bool(score_code_heldout(task, code, backend=backend).get("passed_all"))
        else:
            from macs.analysis.heuristic_score import score_heuristic_code
            r = score_heuristic_code(task, code, backend=backend)
            row["correct"] = bool(r["valid"])
            row["vs_baseline"] = r["score"]
        row["novelty_z"] = None; row["novel"] = False
        if ref is not None:
            X = ast_features(normalise(code)).reshape(1, -1)
            row["novelty_z"] = round(float(ref.novelty(X)[0]), 2)
            row["novel"] = bool(ref.percentile(X)[0] >= NOVELTY_PERCENTILE)
        # Arush - ONE combined creativity flag: a proposal is 'creative' only when it is BOTH novel AND
        # correct. This is how the two separate columns (novelty, correctness) become a single answer.
        row["creative"] = bool(row.get("correct") and row.get("novel"))
        rows.append(row)
    return rows


def score_all(backend: str = "subprocess") -> tuple[list[dict], dict]:
    from macs.analysis.aggregate import analyse_dir
    references = {}
    try:
        from macs.metrics.reference import build_reference_stats, REFERENCE_ROOT
        for f in Path(REFERENCE_ROOT).glob("*.jsonl"):
            try:
                references[f.stem] = build_reference_stats(f.stem)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return analyse_dir(str(TRACES_DIR), backend=backend, references=references)


def studio_vs_tutor(summary: dict) -> dict:
    """Average families for studio vs tutor conditions - the one-line headline."""
    def avg(group, key):
        vals = [summary[c][key] for c in group if c in summary and summary[c].get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None
    return {"studio_families": avg(STUDIO, "avg_families"), "tutor_families": avg(TUTOR, "avg_families")}


def run_session(task_dir: str, condition: str, seed: int, backend: str = "subprocess") -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_session.py"),
           "--task", task_dir, "--condition", condition, "--seed", str(seed), "--sandbox", backend]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)


# ----------------------------------------------------------------- Streamlit UI

def main() -> None:  # pragma: no cover - layout isn't unit-tested
    import pandas as pd
    import streamlit as st
    from macs.tasks.bank import load_bank

    st.set_page_config(page_title="MACS dashboard", layout="wide")
    st.title("Multi-Agent Coding Studio - dashboard")
    st.caption("Measuring whether a studio of agents writes more creative, still-correct code than a single tutor.")

    tab_over, tab_cre, tab_tr, tab_run = st.tabs(
        ["📊 Overview", "🎨 How creativity is measured", "🔍 Traces", "▶ Run"])

    # ============================================================ OVERVIEW
    with tab_over:
        left, right = st.columns([3, 1])
        left.subheader("Studio vs Tutor - the comparison")
        backend = right.selectbox("Sandbox", ["subprocess", "docker"], help="subprocess = dev; docker = isolated")
        if st.button("Re-score all traces", type="primary"):
            with st.spinner("scoring traces..."):
                st.session_state["scored"] = score_all(backend)
        scored = st.session_state.get("scored")
        if scored is None and RESULTS_FILE.exists():
            data = json.loads(RESULTS_FILE.read_text())
            scored = (data.get("sessions", []), data.get("summary", {}))
        if not scored:
            st.info("No results yet. Click **Re-score all traces**, or run a session in the Run tab first.")
        else:
            _sessions, summary = scored
            sv = studio_vs_tutor(summary)
            if sv["studio_families"] and sv["tutor_families"]:
                c1, c2, c3 = st.columns(3)
                c1.metric("Studio - avg families", sv["studio_families"])
                c2.metric("Tutor - avg families", sv["tutor_families"])
                ratio = round(sv["studio_families"] / sv["tutor_families"], 1) if sv["tutor_families"] else "-"
                c3.metric("Studio explores", f"{ratio}× more approaches")
                st.success(f"The studio produced **{sv['studio_families']}** solution families on average "
                           f"vs **{sv['tutor_families']}** for the tutor - the core prediction of the project.")

            # per-condition cards
            st.markdown("##### Per condition")
            cols = st.columns(len([c for c in CONDITIONS if c in summary]) or 1)
            for col, cond in zip(cols, [c for c in CONDITIONS if c in summary]):
                m = summary[cond]
                tag = "🎓 Tutor" if cond in TUTOR else "🧪 Studio"
                col.markdown(f"**{cond}** · {tag}")
                col.caption(COND_DESC[cond])
                col.metric("Families", m.get("avg_families"))
                col.metric("Novelty (z)", m.get("avg_mean_novelty"))
                col.metric("Correct & novel", m.get("avg_correct_and_novel"))

            # charts
            rows = [{"Condition": c, "Families": summary[c].get("avg_families"),
                     "Correct&Novel": summary[c].get("avg_correct_and_novel"),
                     "Novelty": summary[c].get("avg_mean_novelty")}
                    for c in CONDITIONS if c in summary]
            df = pd.DataFrame(rows).set_index("Condition")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Solution families by condition**")
                st.bar_chart(df["Families"].dropna(), height=260)
                st.caption("The live creativity signal: tutors ≈1, studios ≈6.")
            with cc2:
                st.markdown("**Creativity score (correct & novel)**")
                st.bar_chart(df["Correct&Novel"].dropna(), height=260)
                st.caption("Share of proposals that are both novel and correct (0-1).")

            with st.expander("How to read this (important on the current pilot)"):
                st.markdown(
                    "- **Families is the clearest signal right now**: tutor conditions produce 1, studio "
                    "conditions produce ~6.\n"
                    "- **Novelty and 'correct & novel' are saturated** (≈1.0 everywhere) on this pilot because "
                    "the local model's reference set is very homogeneous - only ~2 distinct shapes - so almost "
                    "everything looks novel next to it. They become discriminating with a richer reference set "
                    "and harder tasks.\n"
                    "- **D vs D_minus** are tied here because this task is easy enough that the studio already "
                    "spans the space without the Facilitator; its effect shows on harder tasks.")
            st.dataframe(
                pd.DataFrame([{"Condition": c, "Setup": COND_DESC[c], "Sessions": summary[c]["n_sessions"],
                               "Families": summary[c].get("avg_families"),
                               "Novelty(z)": summary[c].get("avg_mean_novelty"),
                               "Correct": summary[c].get("avg_correct"),
                               "Correct&Novel": summary[c].get("avg_correct_and_novel"),
                               "Heuristic": summary[c].get("avg_heuristic_score")}
                              for c in CONDITIONS if c in summary]),
                hide_index=True, use_container_width=True)

    # ============================================================ CREATIVITY EXPLAINED
    with tab_cre:
        st.subheader("How creativity is measured")
        st.markdown(
            "Creativity here is **not one number** - that would hide what's happening. It's measured as "
            "three things that are then combined into one headline score. Higher is better for all of them.")
        for key in ("families", "novelty", "correct", "creativity"):
            label, desc, _ = METRICS[key]
            with st.container():
                st.markdown(f"#### {label}")
                st.write(desc)
        st.divider()
        st.markdown("##### How the numbers are produced")
        st.markdown(
            "1. **Represent** each solution as a vector of structural features from its code (loops, "
            "recursion, data structures, ...).\n"
            "2. **Diversity**: cluster a session's proposals into *families* (within-sample standardised "
            "features, Euclidean distance). Count the families.\n"
            "3. **Novelty**: distance from a **frozen reference set** of ordinary solutions, z-scored. The "
            "studio's own outputs are never in that set, so it can't be gamed.\n"
            "4. **Correctness**: run each solution on **held-out** tests it never saw (Tier 1/2), or score it "
            "against a baseline heuristic (Tier 3).\n"
            "5. **Creativity score** = fraction of proposals that are both **novel** (top decile) **and correct**.")
        st.info("Design choice: metrics are kept separate and only joined at the end, so you can always see "
                "*why* a condition scored the way it did. Every threshold is pre-registered before results are "
                "seen (see docs/PREREGISTRATION.md).")

    # ============================================================ TRACES
    with tab_tr:
        st.subheader("Open a session")
        traces = list_traces()
        if not traces:
            st.info("No traces in data/traces/ yet.")
        else:
            import pandas as pd
            pick = st.selectbox("Session", traces, format_func=lambda p: p.name)
            events = read_events(pick)
            h = trace_header(events)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Task", h["task_id"])
            c2.metric("Condition", h["condition"])
            c3.metric("Proposals", len(strategies(events)) or 0)
            c4.metric("Interventions", len(interventions(events)))
            strat = strategies(events)
            if strat:
                st.write("**Strategies proposed:** " + " , ".join(f"`{s}`" for s in strat))
            iv = interventions(events)
            if iv:
                st.write("**Facilitator interventions:**")
                for e in iv:
                    st.caption("after proposal: " + e.get("content", "")[:160])

            # per-proposal results, shown automatically (scored once per session, then cached)
            st.markdown("##### Each proposal, scored")
            st.caption("Does it pass the hidden tests (correct), how unusual is it (novelty), and is it "
                       "'creative' = novel AND correct? The two columns combine into the single 'creative' flag.")
            cache = st.session_state.setdefault("ptable_cache", {})
            if pick.name not in cache:
                with st.spinner("scoring each solution in the sandbox (no model needed)..."):
                    try:
                        cache[pick.name] = proposal_table(pick, backend="subprocess")
                    except Exception as ex:  # noqa: BLE001
                        st.error("Could not score this session: " + str(ex)[:200]); cache[pick.name] = []
            tbl = cache[pick.name]
            if tbl:
                is_heur = "vs_baseline" in tbl[0]
                cols = ["proposal", "strategy", "lines", "correct",
                        ("vs_baseline" if is_heur else "novelty_z"), "creative"]
                show = [{c: r.get(c) for c in cols} for r in tbl]
                st.dataframe(pd.DataFrame(show), hide_index=True, use_container_width=True)
                st.write("**The code behind each proposal:**")
                for r in tbl:
                    verdict = "correct" if r.get("correct") else "wrong"
                    with st.expander(f"{r['proposal']}  |  {r['strategy']}  |  {verdict}"):
                        st.code(r["code"], language="python")
            if st.button("Re-score this session"):
                cache.pop(pick.name, None)
                st.rerun()
            with st.expander("Full event timeline"):
                st.dataframe(pd.DataFrame(event_rows(events)), hide_index=True, use_container_width=True)

    # ============================================================ RUN
    with tab_run:
        st.subheader("Run new sessions")
        st.write("Starts sessions with the local model. **Ollama must be running** (the Run button fails "
                 "with a connection error otherwise).")
        tasks = {t.id: t for t in load_bank()}
        tiers = sorted({t.tier for t in tasks.values()})
        tier = st.selectbox("Tier", ["all"] + [f"Tier {t}" for t in tiers])
        ids = sorted(tid for tid, t in tasks.items() if tier == "all" or f"Tier {t.tier}" == tier)
        task_id = st.selectbox("Task", ids)
        colc, cols, colb = st.columns(3)
        seed = cols.number_input("Seed", min_value=1, value=1, step=1)
        backend = colb.selectbox("Sandbox", ["subprocess", "docker"])
        run_all = st.checkbox("Run all five conditions (A, B, C, D, D_minus)")
        condition = colc.selectbox("Condition", CONDITIONS, format_func=lambda c: f"{c} - {COND_DESC[c]}",
                                   disabled=run_all)
        if st.button("Run", type="primary"):
            task_dir = str(tasks[task_id].path)
            targets = CONDITIONS if run_all else [condition]
            for cond in targets:
                with st.spinner(f"running {task_id} / {cond}..."):
                    try:
                        res = run_session(task_dir, cond, int(seed), backend)
                        if res.returncode == 0:
                            st.success(f"{cond}: done")
                            st.code(res.stdout[-1500:] or "(no output)")
                        else:
                            st.error(f"{cond}: failed")
                            st.code(res.stderr[-1200:])
                            break
                    except subprocess.TimeoutExpired:
                        st.error(f"{cond}: timed out")
                        break
            st.info("Now open **Overview** and click *Re-score all traces* to include these runs.")


if __name__ == "__main__":
    main()
