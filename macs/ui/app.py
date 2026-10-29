"""MACS dashboard: the whole pipeline without the terminal.

Tabs:
  * Overview    - the Studio-vs-Tutor comparison; "Analyse all traces" scores every current trace.
  * Statistics  - S8: mixed-effects contrasts with 95% CIs, Holm, minimum detectable effect.
  * Creativity  - how creativity is measured: diversity, novelty, correctness, correct-and-novel.
  * Traces      - open any session: proposals, code, correctness, novelty, interventions.
  * References  - per-task reference status, clear old references, generate new ones, and the
                  Facilitator firing check.
  * Multi-run   - tasks x conditions x seeds (number of seeds chosen here), in the background,
                  with progress and a Stop button; plus a single quick session.
  * Labelling   - SC3 blind hand-labelling (two raters, agreement) and threshold calibration.
  * Setup       - model server, model, torch, Docker and statistics checks.

Model profile and sandbox are chosen once in the sidebar. Long steps run as background jobs
(macs/ui/jobs.py), so refreshing or closing the page does not stop them.

Run with:   streamlit run macs/ui/app.py

Data-loading logic is kept in plain functions (here and in macs/ui/pipeline.py) so it can be
tested without Streamlit; only main() imports `st`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "data" / "traces"
ARCHIVE_DIR = ROOT / "data" / "traces_archive"
RESULTS_FILE = ROOT / "data" / "results" / "summary.json"
NOVELTY_PERCENTILE = 0.90

from macs import conditions as _conds  # noqa: E402

CONDITIONS = _conds.ALL
COND_DESC = {c: v.description for c, v in _conds.CONDITIONS.items()}
TUTOR = set(_conds.TUTOR)
STUDIO = set(_conds.STUDIO)

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


def trace_index(paths: list[Path]) -> list[dict]:
    """Model, task, condition and seed of each trace, from its first line only (fast even for
    thousands of traces)."""
    out = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                first = json.loads(fh.readline())
        except (OSError, ValueError):
            continue
        h = first.get("header", {})
        out.append({"path": p, "model": h.get("model", "?"), "task_id": first.get("task_id", "?"),
                    "condition": first.get("condition", "?"), "seed": h.get("seed", "?")})
    return out


def archive_traces(traces_dir: Path = TRACES_DIR, archive_root: Path = ARCHIVE_DIR,
                   stamp: str | None = None) -> tuple[Path | None, int]:
    """Move every trace in traces_dir into archive_root/<stamp>/ and return (folder, count).

    Traces are moved rather than deleted so a clear can be undone by moving the files back.
    Returns (None, 0) when there is nothing to move.
    """
    from datetime import datetime
    files = sorted(Path(traces_dir).glob("*.jsonl"))
    if not files:
        return None, 0
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = Path(archive_root) / stamp
    n = 1
    while dest.exists():
        dest = Path(archive_root) / f"{stamp}-{n}"
        n += 1
    dest.mkdir(parents=True)
    for f in files:
        f.replace(dest / f.name)
    return dest, len(files)


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
        # a proposal is 'creative' only when it is both novel and correct
        row["creative"] = bool(row.get("correct") and row.get("novel"))
        rows.append(row)
    return rows


def score_all(backend: str = "subprocess") -> tuple[list[dict], dict]:
    from macs.analysis.aggregate import analyse_dir, resolve_model
    from macs.metrics.reference import load_references
    target = resolve_model(str(TRACES_DIR))
    references = load_references(target, log=None) if target else {}
    return analyse_dir(str(TRACES_DIR), backend=backend, references=references)


def studio_vs_tutor(summary: dict) -> dict:
    """Average families for studio vs tutor conditions - the one-line headline.

    Uses families at a matched number of solutions per task ('avg_families_matched') when
    available. The per-session count is capped at 1 for a tutor by construction, so it is only a
    fallback for old summary files.
    """
    def avg(group, key):
        vals = [summary[c][key] for c in group if c in summary and summary[c].get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None
    key = "avg_families_matched" if any(summary.get(c, {}).get("avg_families_matched") is not None
                                        for c in CONDITIONS) else "avg_families"
    return {"studio_families": avg(STUDIO, key), "tutor_families": avg(TUTOR, key), "matched": key != "avg_families"}


def run_session(task_dir: str, condition: str, seed: int, backend: str = "subprocess",
                profile: str = "local") -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_session.py"), "--task", task_dir,
           "--condition", condition, "--seed", str(seed), "--sandbox", backend, "--profile", profile]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)


# ----------------------------------------------------------------- Streamlit UI

def _job_panel(st, kind: str, title: str, *, total_hint: int | None = None) -> dict:
    """Status of a background job: progress, last lines of output, and a Stop button."""
    from macs.ui import jobs
    stt = jobs.status(kind)
    state = stt["state"]
    if state == "none":
        return stt
    meta = stt["meta"]
    icon = {"running": "⏳", "finished": "✅", "failed": "❌", "stopped": "⏹"}.get(state, "")
    st.markdown(f"**{icon} {title}: {state}**  ·  started {meta.get('started', '?')}"
                + (f"  ·  ended {meta['ended']}" if meta.get("ended") else ""))
    prog = stt["progress"]
    if prog:
        k, n = prog
        st.progress(min(k / max(n, 1), 1.0), text=f"{k} of {n}")
    elif state == "running" and total_hint:
        st.caption("starting...")
    if state == "running":
        if st.button("Stop", key=f"stop_{kind}"):
            jobs.stop(kind)
            st.rerun()
    with st.expander("Output", expanded=state in ("failed",)):
        st.code(stt["log"] or "(no output yet)", language="text")
    return stt


def _live(st, key: str, fn, every: int = 3):
    """Run fn inside an auto-refreshing fragment when Streamlit supports it (1.37+); otherwise
    once, with a Refresh button. `key` gives each live panel its own identity."""
    def panel():
        fn()
    panel.__name__ = panel.__qualname__ = f"live_{key}"
    frag = getattr(st, "fragment", None)
    if frag is None:
        panel()
        if st.button("Refresh", key=f"refresh_{key}"):
            st.rerun()
        return
    frag(run_every=every)(panel)()


def main() -> None:  # pragma: no cover - layout isn't unit-tested
    import pandas as pd
    import streamlit as st
    from macs.tasks.bank import load_bank
    from macs.ui import jobs, pipeline

    st.set_page_config(page_title="MACS dashboard", layout="wide")
    st.title("Multi-Agent Coding Studio - dashboard")
    st.caption("Measuring whether a studio of agents writes more creative, still-correct code than a single tutor.")

    # ---------------------------------------------------------- sidebar: shared settings
    profiles = pipeline.list_profiles()
    names = [p["name"] for p in profiles] or ["local"]
    default = next((i for i, p in enumerate(profiles) if p["default"]), 0)
    with st.sidebar:
        st.header("Settings")
        profile = st.selectbox("Model profile", names, index=default,
                               help="From config/models.yaml. References, runs and analysis use this model.")
        prof = next((p for p in profiles if p["name"] == profile), {})
        model = prof.get("model") or None
        st.caption(f"model: `{model or '?'}`  \nrevision: `{prof.get('revision') or '-'}`")
        sandbox = st.selectbox("Sandbox", ["subprocess", "docker"],
                               help="subprocess for development; docker (isolated) for numbers you report")
        st.divider()
        st.markdown("**Background jobs**")
        any_job = False
        for kind, label in [("references", "References"), ("multirun", "Multi-run"), ("analyze", "Analysis"),
                            ("stats", "Statistics"), ("tau", "Facilitator check"), ("calibrate", "SC3 calibration")]:
            s_ = jobs.status(kind)["state"]
            if s_ != "none":
                any_job = True
                st.caption(f"{label}: {s_}")
        if not any_job:
            st.caption("none yet")

    bank = load_bank()
    task_by_id = {t.id: t for t in bank}

    tabs = st.tabs(["📊 Overview", "📈 Statistics", "🎨 How creativity is measured", "🔍 Traces",
                    "📚 References", "🚀 Multi-run", "🏷 Labelling (SC3)", "🧑‍⚖️ Rating study (SC4)",
                    "🧩 Open problems", "⚙ Setup"])
    (tab_over, tab_stats, tab_cre, tab_tr, tab_ref, tab_multi, tab_lab, tab_rating, tab_open,
     tab_setup) = tabs
    from macs.ui import extra_tabs
    ctx = {"pd": pd, "jobs": jobs, "ROOT": ROOT, "model": model, "profile": profile, "sandbox": sandbox,
           "bank": bank, "job_panel": lambda kind, title: _job_panel(st, kind, title),
           "live": lambda key, fn: _live(st, key, fn)}

    # ============================================================ OVERVIEW
    with tab_over:
        st.subheader("Studio vs Tutor - the comparison")
        st.caption("Every number here is for ONE model; results from different models are never pooled "
                   "(the Statistics tab compares models). The main model is Qwen3.5-27B.")
        # one summary per model: the main file (27B) plus data/results/models/<model>/summary.json
        _shown = {}
        for _f in [RESULTS_FILE, *sorted((RESULTS_FILE.parent / "models").glob("*/summary.json"))]:
            try:
                _m = json.loads(_f.read_text(encoding="utf-8")).get("model") if _f.exists() else None
            except (OSError, ValueError):
                _m = None
            if _m and _m not in _shown:
                _shown[_m] = _f
        view_model = st.selectbox("Model shown", list(_shown) or ["(no results yet)"], key="ov_model",
                                  format_func=lambda m: m + ("  (main model)" if m == "qwen3.5:27b" else ""))
        rep_view = st.radio("Representation", ["AST (main analysis)", "UniXcoder (parallel, needs torch)"],
                            horizontal=True, key="ov_rep")
        rep_arg = "unixcoder" if rep_view.startswith("UniXcoder") else "ast"
        results_file = _shown.get(view_model, RESULTS_FILE) if rep_arg == "ast" else \
            RESULTS_FILE.with_name("summary_unixcoder.json")
        if rep_arg != "ast":
            st.caption("The UniXcoder view exists for the main model only.")
        if st.button(f"Re-analyse {view_model}'s traces", type="primary", key="analyze_btn",
                     help="Rescores this model's traces on this computer and rewrites its summary (slow with "
                          "thousands of sessions; the ISCA analysis job does all models at once)."):
            try:
                out = str(results_file)
                args = ["--sandbox", sandbox, "--representation", rep_arg, "--model", view_model, "--out", out]
                jobs.start("analyze", "analyze.py", args)
                st.rerun()
            except RuntimeError as ex:
                st.warning(str(ex))
        _live(st, "analyze", lambda: _job_panel(st, "analyze", "Analysis"))
        scored = None
        if results_file.exists():
            data = json.loads(results_file.read_text(encoding="utf-8"))
            scored = (data.get("sessions", []), data.get("summary", {}))
            st.markdown(f"**Model: `{data.get('model', '?')}`** · {len(scored[0])} sessions")
        if not scored or not scored[1]:
            st.info("No results yet. Run sessions in **Multi-run**, then **Analyse all traces**.")
        else:
            _sessions, summary = scored
            sv = studio_vs_tutor(summary)
            if sv["studio_families"] and sv["tutor_families"]:
                c1, c2, c3 = st.columns(3)
                c1.metric("Studio - avg families", sv["studio_families"])
                c2.metric("Tutor - avg families", sv["tutor_families"])
                ratio = round(sv["studio_families"] / sv["tutor_families"], 1) if sv["tutor_families"] else "-"
                c3.metric("Studio explores", f"{ratio}× more approaches")
                basis = ("at the same number of solutions per task" if sv.get("matched")
                         else "per session (not size-matched: a tutor session has only one solution)")
                st.info(f"Studio: **{sv['studio_families']}** solution families on average vs "
                        f"**{sv['tutor_families']}** for the tutor, {basis}.")
            notes = extra_tabs.conclusion(summary)
            if notes:
                st.success(" ".join(notes))
            st.markdown("##### Per condition")
            present = [c for c in CONDITIONS if c in summary]
            groups = [("🎓 Tutor (one agent)", ["A", "C4", "C"]),
                      ("🧪 Studio (team of agents)", ["B", "D4", "D", "D_minus", "E", "E_minus"]),
                      ("⚖ Baselines (same budget / one prompt)", ["S", "P"])]
            missing = [c for c in CONDITIONS if c not in summary]
            if missing:
                st.caption("Not run for this model: " + ", ".join(missing))
            per_row = 3
            for title, conds in groups:
                conds = [c for c in conds if c in summary]
                if not conds:
                    continue
                st.markdown(f"**{title}**")
                for i in range(0, len(conds), per_row):
                    cols = st.columns(per_row)
                    for col, cond in zip(cols, conds[i:i + per_row]):
                        m = summary[cond]
                        with col.container(border=True):
                            st.markdown(f"#### {cond}")
                            st.caption(COND_DESC[cond] + f" · {m.get('n_sessions', '?')} sessions")
                            def _f(v):
                                return "-" if v is None else f"{v:.2f}"
                            vals = [("Families", _f(m.get("avg_families_matched", m.get("avg_families")))),
                                    ("Novelty z", _f(m.get("avg_mean_novelty"))),
                                    ("Correct & novel", _f(m.get("avg_correct_and_novel")))]
                            st.markdown(
                                "<div style='display:flex;gap:6px;justify-content:space-between'>" + "".join(
                                    f"<div style='flex:1;min-width:0'><div style='font-size:0.72rem;opacity:0.7'>{k}</div>"
                                    f"<div style='font-size:1.35rem;font-weight:600'>{v}</div></div>" for k, v in vals)
                                + "</div>", unsafe_allow_html=True)
            rows = [{"Condition": c, "Families": summary[c].get("avg_families_matched", summary[c].get("avg_families")),
                     "Correct&Novel": summary[c].get("avg_correct_and_novel")} for c in present]
            df = pd.DataFrame(rows).set_index("Condition")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Solution families by condition**")
                st.bar_chart(df["Families"].dropna(), height=260)
                st.caption("Counted at the same number of solutions per task, so tutor and studio are compared fairly.")
            with cc2:
                st.markdown("**Creativity score (correct & novel)**")
                st.bar_chart(df["Correct&Novel"].dropna(), height=260)
                st.caption("Share of proposals that are both novel and correct (0-1).")
            with st.expander("How to read this"):
                st.markdown(
                    "- **Families** are counted at the same number of solutions for every condition on a task. "
                    "Each seed gives a tutor one more solution, so run several seeds per condition.\n"
                    "- **Novelty** is in standard deviations from the task's reference set of ordinary solutions "
                    "(0 = typical). Tasks without a reference set have no novelty.\n"
                    "- **Pass rate and tokens** are the cost side: report them beside every gain.\n"
                    "- **D vs D_minus** differ only in the Facilitator; the Interventions column shows how often "
                    "it fired. Effects with confidence intervals are in the **Statistics** tab.")
            st.dataframe(
                pd.DataFrame([{"Condition": c, "Setup": COND_DESC[c], "Sessions": summary[c]["n_sessions"],
                               "Families (matched)": summary[c].get("avg_families_matched"),
                               "Families/session": summary[c].get("avg_families"),
                               "Novelty(z)": summary[c].get("avg_mean_novelty"),
                               "Correct/session": summary[c].get("avg_correct"),
                               "Correct&Novel": summary[c].get("avg_correct_and_novel"),
                               "Interventions": summary[c].get("avg_interventions"),
                               "Tokens": summary[c].get("avg_tokens")} for c in present]),
                hide_index=True, use_container_width=True)

    # ============================================================ STATISTICS (S8)
    with tab_stats:
        st.subheader("Statistics: planned contrasts (S8)")
        st.caption("Mixed-effects model per outcome (studio x learners, task as random effect), contrasts "
                   "B-A, D-C, the Facilitator (D-D_minus, E-E_minus) and the baselines (D-S, D-P) with 95% CIs, "
                   "Holm across the primary contrasts that were run, and the "
                   "minimum detectable effect for a planned design. Uses data/results/summary.json, so "
                   "run the analysis first.")
        c1, c2 = st.columns(2)
        plan_tasks = c1.number_input("Tasks in the planned main run", 2, 200, 24)
        plan_seeds = c2.number_input("Seeds per cell in the planned main run", 1, 50, 6)
        task_set = st.radio("Task set", ["main", "contamination_check", "all"], horizontal=True, key="st_set",
                            help="main = the 24-task bank (pre-registered); contamination_check = the EvoEval "
                                 "rewrites, analysed on their own")
        if st.button("Compute statistics", type="primary", disabled=not RESULTS_FILE.exists()):
            try:
                jobs.start("stats", "stats_report.py", ["--tasks", str(plan_tasks), "--seeds", str(plan_seeds),
                                                        "--task-set", task_set])
                st.rerun()
            except RuntimeError as ex:
                st.warning(str(ex))
        _live(st, "stats", lambda: _job_panel(st, "stats", "Statistics"))
        stats_file = ROOT / "data" / "results" / "stats.json"
        if stats_file.exists():
            res = json.loads(stats_file.read_text(encoding="utf-8"))

            def table(items, holm_col):
                rows = []
                for c in items:
                    r = {"contrast": c["label"], "estimate": round(c["estimate"], 3),
                         "95% CI": f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]", "p": round(c["p"], 4)}
                    if holm_col:
                        r["p (Holm)"] = None if c.get("p_holm") is None else round(c["p_holm"], 4)
                    rows.append(r)
                return pd.DataFrame(rows)
            st.markdown(f"**Primary (Holm family)** · task set: {res.get('task_set', 'main')}")
            st.dataframe(table(res.get("primary", []), True), hide_index=True, use_container_width=True)
            st.markdown("**Secondary (exploratory, unadjusted)**")
            st.dataframe(table(res.get("secondary", []), False), hide_index=True, use_container_width=True)
            d = res.get("design", {})
            st.markdown(f"**Minimum detectable effect** for {d.get('tasks')} tasks x {d.get('seeds')} seeds "
                        "(80% power, Holm-strict alpha)")
            st.dataframe(pd.DataFrame([{"outcome": k, "MDE": round(v["mde"], 3),
                                        "residual SD": round(v["sigma2_residual"] ** 0.5, 3)}
                                       for k, v in res.get("mde", {}).items()]),
                         hide_index=True, use_container_width=True)

        extra_tabs.statistics_extras(st, ctx)

    # ============================================================ CREATIVITY EXPLAINED
    with tab_cre:
        st.subheader("How creativity is measured")
        st.markdown(
            "Creativity here is **not one number** - that would hide what's happening. It's measured as "
            "three things that are then combined into one headline score. Higher is better for all of them.")
        for key in ("families", "novelty", "correct", "creativity"):
            label, desc, _ = METRICS[key]
            st.markdown(f"#### {label}")
            st.write(desc)
        st.divider()
        st.markdown("##### How the numbers are produced")
        st.markdown(
            "1. **Represent** each solution as a vector of structural features from its code (loops, "
            "recursion, data structures, ...).\n"
            "2. **Diversity**: cluster the solutions into *families* (standardised features, Euclidean "
            "distance), counted at the same number of solutions for every condition.\n"
            "3. **Novelty**: distance from a **frozen reference set** of ordinary solutions, z-scored. The "
            "studio's own outputs are never in that set.\n"
            "4. **Correctness**: run each solution on **held-out** tests it never saw (Tier 1/2), or score it "
            "against a baseline heuristic (Tier 3).\n"
            "5. **Creativity score** = fraction of proposals that are both **novel** (top decile) **and correct**.")
        st.info("Metrics are kept separate and only joined at the end, so you can always see why a condition "
                "scored the way it did. Every threshold is pre-registered (docs/PREREGISTRATION.md).")

    # ============================================================ TRACES
    with tab_tr:
        st.subheader("Open a session")
        traces = list_traces()
        if traces:
            with st.expander(f"Clear all traces ({len(traces)} in data/traces/)"):
                st.caption("Moves every trace into data/traces_archive/<date-time>/. Nothing is deleted; "
                           "move the files back into data/traces/ to restore them.")
                confirm = st.checkbox("I want to clear all traces", key="confirm_clear")
                if st.button("Clear all traces", type="primary", disabled=not confirm):
                    dest, moved = archive_traces()
                    for k in ("ptable_cache", "confirm_clear"):
                        st.session_state.pop(k, None)
                    st.session_state["cleared_msg"] = f"Moved {moved} trace(s) to {dest.relative_to(ROOT)}"
                    st.rerun()
        if "cleared_msg" in st.session_state:
            st.success(st.session_state.pop("cleared_msg"))
        if not traces:
            st.info("No traces in data/traces/ yet.")
        else:
            @st.cache_data(show_spinner="indexing traces by model...")
            def _index(names: tuple[str, ...]) -> list[dict]:
                return [{**r, "path": str(r["path"])} for r in trace_index([TRACES_DIR / n for n in names])]
            idx = _index(tuple(p.name for p in traces))
            models = sorted({r["model"] for r in idx})
            f1, f2, f3 = st.columns(3)
            tm = f1.selectbox("Model", models, key="tr_model",
                              index=models.index("qwen3.5:27b") if "qwen3.5:27b" in models else 0,
                              format_func=lambda m: m + ("  (main)" if m == "qwen3.5:27b" else ""))
            mine = [r for r in idx if r["model"] == tm]
            tt = f2.selectbox("Task", ["all"] + sorted({r["task_id"] for r in mine}), key="tr_task")
            tc = f3.selectbox("Condition", ["all"] + sorted({r["condition"] for r in mine}), key="tr_cond")
            mine = [r for r in mine if (tt == "all" or r["task_id"] == tt) and (tc == "all" or r["condition"] == tc)]
            st.caption(f"{len(mine)} session(s) of {tm} match; each model's sessions are separate.")
            if not mine:                                   # never st.stop(): later tabs must still render
                st.info("No session matches these filters; showing all of this model's sessions.")
                mine = [r for r in idx if r["model"] == tm]
            pick = Path(st.selectbox("Session", [r["path"] for r in mine], key="tr_pick",
                                     format_func=lambda p: Path(p).name))
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
            for e in interventions(events):
                st.caption("Facilitator: " + e.get("content", "")[:200])
            st.markdown("##### Each proposal, scored")
            st.caption("Does it pass the hidden tests (correct), how unusual is it (novelty), and is it "
                       "'creative' = novel AND correct?")
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
                cols = ["proposal", "strategy", "lines", "correct", ("vs_baseline" if is_heur else "novelty_z"), "creative"]
                st.dataframe(pd.DataFrame([{c: r.get(c) for c in cols} for r in tbl]),
                             hide_index=True, use_container_width=True)
                st.write("**The code behind each proposal:**")
                for r in tbl:
                    verdict = "correct" if r.get("correct") else "wrong"
                    with st.expander(f"{r['proposal']}  |  {r['strategy']}  |  {verdict}"):
                        st.code(r["code"], language="python")
                        task_obj = task_by_id.get(h["task_id"])
                        if task_obj is not None and task_obj.scoring == "tests":
                            outs = pipeline.example_outputs(task_obj, r["code"], backend=sandbox)
                            if outs:
                                st.caption("What it returns for the example inputs:")
                                st.code("\n".join(f"{call}  ->  {val}" for call, val in outs), language="text")
            if st.button("Re-score this session"):
                cache.pop(pick.name, None)
                st.rerun()
            with st.expander("Full event timeline"):
                st.dataframe(pd.DataFrame(event_rows(events)), hide_index=True, use_container_width=True)

    # ============================================================ REFERENCES
    with tab_ref:
        st.subheader("Reference sets (ordinary solutions per task)")
        st.caption("Each task needs a frozen set of ordinary solutions from the experiment's model: novelty is "
                   "measured against it and the Facilitator's threshold comes from it. Needs the model server.")
        target = st.number_input("Samples per task", 10, 500, 100, step=10,
                                 help="The proposal uses 100; 50 and 200 are for the sensitivity check.")

        def ref_status():
            rows = pipeline.reference_table(bank, model=model, target=int(target))
            counts = {s: sum(r["state"] == s for r in rows) for s in ("ready", "incomplete", "missing", "other model")}
            c = st.columns(4)
            for col, (k, v) in zip(c, counts.items()):
                col.metric(k.capitalize(), v)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            _job_panel(st, "references", "Reference build")
        _live(st, "references", ref_status, every=5)

        rows = pipeline.reference_table(bank, model=model, target=int(target))
        other = [r["task"] for r in rows if r["state"] == "other model"]
        st.markdown("##### Clear old references")
        st.caption("Moves reference files to data/reference_archive/<date-time>/ (nothing is deleted). Needed "
                   "before building with a different model: a reference set must come from one model.")
        existing = [r["task"] for r in rows if r["samples"] > 0]
        which = st.radio("Which", ["Only those from another model", "All", "Choose"], horizontal=True, key="ref_clear_which")
        if which == "Choose":
            chosen = st.multiselect("Tasks to clear", existing, key="ref_clear_pick")
        else:
            chosen = other if which.startswith("Only") else existing
        st.caption(f"{len(chosen)} file(s) selected")
        ok = st.checkbox("I want to clear these reference files", key="ref_clear_ok")
        if st.button("Clear references", disabled=not (ok and chosen)):
            if jobs.status("references")["state"] == "running":
                st.warning("A reference build is running; stop it first.")
            else:
                dest, n = pipeline.archive_references(chosen)
                st.success(f"Moved {n} file(s) to {dest.relative_to(ROOT)}")
                st.rerun()

        st.markdown("##### Generate references")
        mode = st.radio("Tasks", ["Missing or incomplete only", "All tasks", "Choose"], horizontal=True, key="ref_gen_mode")
        if mode == "Choose":
            gen = st.multiselect("Tasks", [t.id for t in bank], key="ref_gen_pick")
        elif mode == "All tasks":
            gen = [t.id for t in bank]
        else:
            gen = [r["task"] for r in rows if r["state"] in ("missing", "incomplete")]
        blocked = [t for t in gen if t in other]
        if blocked:
            st.warning(f"{len(blocked)} selected task(s) hold references from another model "
                       f"({', '.join(blocked[:4])}{'...' if len(blocked) > 4 else ''}). Clear them first.")
        st.caption(f"{len(gen)} task(s), up to {int(target)} samples each, model `{model}`. Existing samples "
                   "are kept and topped up.")
        if st.button("Generate references", type="primary", disabled=not gen or bool(blocked)):
            try:
                spec = "all" if set(gen) == {t.id for t in bank} else ",".join(gen)
                jobs.start("references", "build_reference.py",
                           ["--tasks", spec, "--n", str(int(target)), "--profile", profile, "--sandbox", sandbox])
                st.rerun()
            except RuntimeError as ex:
                st.warning(str(ex))

        st.markdown("##### Facilitator check (firing rate)")
        st.caption("Pre-registered check that the Facilitator's threshold gives a non-degenerate firing rate, "
                   "on D_minus traces. No model calls. UniXcoder needs torch and transformers.")
        rep = st.radio("Space", ["unixcoder", "ast"], horizontal=True, key="tau_rep")
        from macs.orchestrator import check_facilitator_dependencies
        missing = check_facilitator_dependencies("unixcoder") if rep == "unixcoder" else None
        if missing:
            st.error(missing)
        if st.button("Run Facilitator check", disabled=bool(missing)):
            try:
                jobs.start("tau", "tau_diagnostics.py", ["--representation", rep])
                st.rerun()
            except RuntimeError as ex:
                st.warning(str(ex))
        _live(st, "tau", lambda: _job_panel(st, "tau", "Facilitator check"))
        tfile = ROOT / "data" / "results" / f"tau_diagnostics_{rep}.json"
        if tfile.exists():
            td = json.loads(tfile.read_text(encoding="utf-8"))
            st.dataframe(pd.DataFrame([{"task": r["task_id"], "tau": r["tau_matched_rule"], "checks": r["studio_checks"],
                                        **{f"fire p{k}": v["fire_rate"] for k, v in r.get("by_percentile", {}).items()}}
                                       for r in td.get("tasks", [])]), hide_index=True, use_container_width=True)
        extra_tabs.reference_sensitivity(st, ctx)

    # ============================================================ MULTI-RUN
    with tab_multi:
        st.subheader("Multi-run: tasks x conditions x seeds")
        st.caption("Runs every chosen task under every chosen condition for each seed, in the background. Safe to "
                   "stop and start again: sessions already done with this model are skipped. Needs the model server.")
        scope = st.radio("Tasks", ["All tasks", "Tier 1", "Tier 2", "Tier 3", "Choose"], horizontal=True, key="mr_scope")
        if scope == "Choose":
            run_tasks = st.multiselect("Tasks", [t.id for t in bank], key="mr_pick")
        elif scope == "All tasks":
            run_tasks = [t.id for t in bank]
        else:
            run_tasks = [t.id for t in bank if t.tier == int(scope[-1])]
        conds = st.multiselect("Conditions", CONDITIONS, default=_conds.CORE,
                               format_func=lambda c: f"{c} - {COND_DESC[c]}", key="mr_conds")
        c1, c2 = st.columns(2)
        n_seeds = c1.number_input("Number of seeds", 1, 50, 6, help="The proposal uses 6 repetitions per cell.")
        first_seed = c2.number_input("First seed", 1, 10000, 1)
        seeds = list(range(int(first_seed), int(first_seed) + int(n_seeds)))
        # several models can be run one after another; each is analysed separately (never pooled)
        model_of = {p["name"]: p["model"] for p in profiles}
        run_profiles = st.multiselect("Models (profiles)", list(model_of), default=[profile], key="mr_profiles",
                                      help="Run the same tasks, conditions and seeds on each chosen model, one after "
                                           "another. 'deepseek' is the AI judge, not an experiment model.")
        if run_tasks and conds and run_profiles:
            plans = [pipeline.run_plan(run_tasks, conds, seeds, model_of[p]) for p in run_profiles]
            plan = {k: sum(pl[k] for pl in plans) for k in ("total", "done", "to_run")}
            m1, m2, m3 = st.columns(3)
            m1.metric("Sessions in this run", plan["total"])
            m2.metric("Already done (these models)", plan["done"])
            m3.metric("To run", plan["to_run"])
            for p in run_profiles:
                refs = {r["task"]: r["state"] for r in pipeline.reference_table(bank, model=model_of[p])}
                not_ready = [t for t in run_tasks if refs.get(t) != "ready"]
                if not_ready:
                    st.warning(f"{p}: {len(not_ready)} task(s) have no ready reference for `{model_of[p]}`, so they "
                               f"run without novelty and with the default Facilitator threshold: "
                               f"{', '.join(not_ready[:5])}{'...' if len(not_ready) > 5 else ''}. Build references "
                               f"for this model first (References tab, with this profile chosen in the sidebar).")
            if any(c in STUDIO for c in conds):
                from macs.orchestrator import check_facilitator_dependencies
                problem = check_facilitator_dependencies()
                if problem:
                    st.error(problem)
            st.caption(f"Models {', '.join(f'`{model_of[p]}`' for p in run_profiles)}, sandbox `{sandbox}`, "
                       f"seeds {pipeline.seeds_spec(int(first_seed), int(n_seeds))}.")
            if st.button("Start multi-run", type="primary", disabled=plan["to_run"] == 0):
                try:
                    spec = "all" if set(run_tasks) == {t.id for t in bank} else ",".join(run_tasks)
                    jobs.start("multirun", "run_experiment.py",
                               ["--tasks", spec, "--conditions", ",".join(conds),
                                "--seeds", pipeline.seeds_spec(int(first_seed), int(n_seeds)),
                                "--profile", ",".join(run_profiles), "--sandbox", sandbox])
                    st.rerun()
                except RuntimeError as ex:
                    st.warning(str(ex))
        _live(st, "multirun", lambda: _job_panel(st, "multirun", "Multi-run"))
        st.info("When the run has finished, open **Overview** and click **Analyse all traces**.")

        with st.expander("Single session (quick check, runs in the page)"):
            task_id = st.selectbox("Task", sorted(task_by_id), key="single_task")
            cc, cs = st.columns(2)
            condition = cc.selectbox("Condition", CONDITIONS, format_func=lambda c: f"{c} - {COND_DESC[c]}", key="single_cond")
            seed = cs.number_input("Seed", min_value=1, value=1, step=1, key="single_seed")
            if st.button("Run one session", key="single_run"):
                with st.spinner(f"running {task_id} / {condition}..."):
                    try:
                        res = run_session(str(task_by_id[task_id].path), condition, int(seed), sandbox, profile)
                        (st.success if res.returncode == 0 else st.error)(f"{condition}: "
                                                                         + ("done" if res.returncode == 0 else "failed"))
                        st.code((res.stdout if res.returncode == 0 else res.stderr)[-1500:] or "(no output)")
                    except subprocess.TimeoutExpired:
                        st.error("timed out")

    # ============================================================ LABELLING (SC3)
    with tab_lab:
        st.subheader("SC3: blind labelling of solution families")
        st.info("Since 25 Oct 2026 the labels come from independent raters in Part B of the rater form "
                "(48 solutions, pre-registered approach lists); the researcher and the supervisor do not "
                "label. Their majority labels go to data/calibration/labels.jsonl, then "
                "`python scripts/calibrate_clustering.py` sets the threshold (target ARI >= 0.60).")
        _items = json.loads((ROOT / "data" / "rating" / "label_items.json").read_text(encoding="utf-8")) \
            if (ROOT / "data" / "rating" / "label_items.json").exists() else []
        _res = json.loads((ROOT / "data" / "rating" / "results.json").read_text(encoding="utf-8")) \
            if (ROOT / "data" / "rating" / "results.json").exists() else {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Part B items", len(_items))
        c2.metric("Human raters", len((_res.get("sc3_labels") or {}).get("raters", [])))
        c3.metric("Fleiss' kappa (humans)", (_res.get("sc3_labels") or {}).get("fleiss_kappa") or "-")
        if _res.get("sc3_labels_judge"):
            st.caption(f"AI rater (reported separately): {_res['sc3_labels_judge']}")
        st.markdown("##### Older in-dashboard labelling tool")
        st.caption("Builds its pool only when asked: it reads every trace, which takes minutes with thousands of sessions.")
        c1, c2 = st.columns(2)
        rater = c1.radio("Rater", ["rater1", "rater2"], horizontal=True,
                         help="rater1's labels calibrate the threshold; rater2 is the second rater for agreement.")
        per_task = c2.number_input("Solutions per task", 5, 500, 40)
        if st.button("Build / refresh the labelling pool"):
            with st.spinner("reading every trace..."):
                st.session_state["lab_pool"] = pipeline.label_pool(per_task=int(per_task))
            st.session_state["lab_i"] = 0
        pool = st.session_state.get("lab_pool", [])
        path = pipeline.labels_path(rater)
        labels = pipeline.load_labels(path)
        if not pool:
            st.info("No pool built (press the button above), or no current traces yet.")
        else:
            tasks_in_pool = sorted({r["task_id"] for r in pool})
            prog = pd.DataFrame([{"task": t, "labelled": sum(1 for r in pool if r["task_id"] == t and r["key"] in labels),
                                  "in pool": sum(1 for r in pool if r["task_id"] == t)} for t in tasks_in_pool])
            st.dataframe(prog, hide_index=True, use_container_width=True)
            task_pick = st.selectbox("Task", tasks_in_pool, key="lab_task")
            items = [r for r in pool if r["task_id"] == task_pick]
            only_open = st.checkbox("Show unlabelled only", value=True, key="lab_open")
            view = [r for r in items if r["key"] not in labels] if only_open else items
            if not view:
                st.success("Every solution of this task is labelled.")
            else:
                i = min(st.session_state.get("lab_i", 0), len(view) - 1)
                item = view[i]
                st.markdown(f"**Solution {i + 1} of {len(view)}**")
                st.code(item.get("display") or item["code"], language="python")
                fams = list(task_by_id[task_pick].known_families) if task_pick in task_by_id else []
                options = fams + [pipeline.OTHER, pipeline.UNSURE]
                current = labels.get(item["key"], {}).get("family")
                # nothing is pre-selected: a label is only saved once the rater has chosen one
                choice = st.radio("Approach", options, index=options.index(current) if current in options else None,
                                  key=f"lab_choice_{item['key']}")
                b1, b2, b3 = st.columns(3)
                if b1.button("Save and next", type="primary", disabled=choice is None):
                    pipeline.save_label(path, item, choice)
                    st.session_state["lab_i"] = i if only_open else i + 1
                    st.rerun()
                if b2.button("Previous", disabled=i == 0):
                    st.session_state["lab_i"] = i - 1
                    st.rerun()
                if b3.button("Skip", disabled=i >= len(view) - 1):
                    st.session_state["lab_i"] = i + 1
                    st.rerun()
                st.caption("`other` = a genuinely different approach not in the list; `unsure` is left out of the "
                           "calibration.")
        st.divider()
        p1, p2 = pipeline.labels_path("rater1"), pipeline.labels_path("rater2")
        if p1.exists() and p2.exists():
            ag = pipeline.rater_agreement(p1, p2)
            st.metric("Agreement between raters (Cohen's kappa)", ag["kappa"] if ag["kappa"] is not None else "-",
                      help=f"on {ag['n']} solutions both raters labelled; raw agreement {ag['agreement']}")
        st.markdown("##### Calibrate the clustering threshold")
        st.caption("Sweeps the threshold and reports the one whose automatic families best match rater1's labels "
                   "(mean ARI over tasks). Record the result in docs/PREREGISTRATION.md and set CLUSTER_THRESHOLD "
                   "in macs/analysis/aggregate.py; it is then frozen.")
        if st.button("Run calibration", disabled=not p1.exists()):
            try:
                jobs.start("calibrate", "calibrate_clustering.py", [])
                st.rerun()
            except RuntimeError as ex:
                st.warning(str(ex))
        _live(st, "calibrate", lambda: _job_panel(st, "calibrate", "SC3 calibration"))

    # ============================================================ RATING STUDY (SC4)
    with tab_rating:
        extra_tabs.rating_study(st, ctx)

    # ============================================================ OPEN PROBLEMS
    with tab_open:
        extra_tabs.open_problems(st, ctx)

    # ============================================================ SETUP
    with tab_setup:
        st.subheader("Setup checks")
        st.dataframe(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in p.items()}
                                   for p in profiles]), hide_index=True, use_container_width=True)
        st.caption("Profiles come from config/models.yaml (API keys are not shown). Edit that file to add a model.")
        if st.button("Run checks"):
            import shutil
            results = []
            try:
                import requests
                r = requests.get(prof.get("base_url", "").rstrip("/") + "/models", timeout=5)
                ids = [m.get("id") for m in r.json().get("data", [])] if r.ok else []
                results.append(("Model server", r.ok, f"{prof.get('base_url')} ({len(ids)} models)"))
                results.append(("Model available", model in ids, model or "-"))
            except Exception as ex:  # noqa: BLE001
                results.append(("Model server", False, f"not reachable: {str(ex)[:80]} (is Ollama running?)"))
            from macs.orchestrator import check_facilitator_dependencies
            problem = check_facilitator_dependencies()
            results.append(("torch + transformers (Facilitator)", problem is None, "installed" if problem is None else "missing"))
            results.append(("Docker (isolated sandbox)", shutil.which("docker") is not None,
                            "found" if shutil.which("docker") else "not found (subprocess sandbox only)"))
            for name in ("statsmodels",):
                import importlib.util
                ok = importlib.util.find_spec(name) is not None
                results.append((name + " (statistics cross-check)", ok, "installed" if ok else "optional"))
            st.dataframe(pd.DataFrame([{"check": a, "ok": "✅" if b else "❌", "detail": c} for a, b, c in results]),
                         hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
