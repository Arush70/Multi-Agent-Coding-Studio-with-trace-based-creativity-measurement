"""Dashboard sections added for the extended study (2026-10-20).

Kept out of app.py so that file stays readable. Every long step runs as a background job, like
the rest of the dashboard. Each function receives the Streamlit module and a context dict:
    pd, jobs, ROOT, model, profile, sandbox, bank, job_panel(kind, title), live(key, fn)
"""
from __future__ import annotations

import json
from pathlib import Path


def _start(st, jobs, kind, script, args):
    try:
        jobs.start(kind, script, args)
        st.rerun()
    except RuntimeError as ex:
        st.warning(str(ex))


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- Overview: plain-language conclusion

def conclusion(summary: dict) -> list[str]:
    """A few sentences that say what the current numbers show, with the caveats that apply."""
    def fam(c):
        v = summary.get(c, {})
        return v.get("avg_families_matched", v.get("avg_families"))

    def cn(c):
        return summary.get(c, {}).get("avg_correct_and_novel")

    out = []
    n = sum(v.get("n_sessions", 0) for v in summary.values())
    tasks_note = f"Based on {n} sessions."
    for studio, tutor, label in (("B", "A", "with one learner"), ("D4", "C4", "with four learners"),
                                 ("D", "C", "with six learners")):
        if fam(studio) is not None and fam(tutor) is not None:
            d = fam(studio) - fam(tutor)
            out.append(f"{label.capitalize()}, the studio found {abs(d):.2f} {'more' if d >= 0 else 'fewer'} "
                       f"solution families than the tutor ({fam(studio):.2f} vs {fam(tutor):.2f}).")
        if cn(studio) is not None and cn(tutor) is not None:
            out.append(f"Its share of proposals that were both correct and novel was {cn(studio):.2f} vs "
                       f"{cn(tutor):.2f}.")
    if fam("D") is not None and fam("D_minus") is not None:
        d = fam("D") - fam("D_minus")
        i = summary.get("D", {}).get("avg_interventions")
        out.append(f"Facilitator on vs off (D vs D_minus): {d:+.2f} families"
                   + (f"; it stepped in {i:.2f} times per session on average." if i is not None else "."))
    if fam("E") is not None and fam("E_minus") is not None:
        out.append(f"With a neutral Innovator (E vs E_minus) the Facilitator changed families by "
                   f"{fam('E') - fam('E_minus'):+.2f}.")
    for base, name in (("S", "repeated sampling at the same token budget"), ("P", "a single diversity prompt")):
        if fam("D") is not None and fam(base) is not None:
            out.append(f"Against {name} ({base}), the studio's family count differs by {fam('D') - fam(base):+.2f}.")
    if out:
        out.append(tasks_note + " Differences here are descriptive; the Statistics tab gives confidence "
                   "intervals and the Holm-adjusted tests.")
    return out


# ---------------------------------------------------------------- Statistics: models, frontier, mutation

def statistics_extras(st, c) -> None:
    pd, jobs, ROOT = c["pd"], c["jobs"], c["ROOT"]
    st.divider()
    st.markdown("##### Is the result the same on every model?")
    st.caption("Analyses each model's traces separately (never pooled), then combines every contrast across "
               "models with a random-effects meta-analysis: pooled estimate, I² (how much the models disagree) "
               "and how many models agree in direction.")
    b1, b2 = st.columns(2)
    if b1.button("1. Analyse every model", key="all_models"):
        _start(st, jobs, "analyze_models", "analyze.py", ["--all-models", "--sandbox", c["sandbox"]])
    if b2.button("2. Compare models", key="consistency"):
        _start(st, jobs, "consistency", "model_consistency.py", [])
    c["live"]("analyze_models", lambda: c["job_panel"]("analyze_models", "Per-model analysis"))
    c["live"]("consistency", lambda: c["job_panel"]("consistency", "Cross-model comparison"))
    rep = _read(ROOT / "data" / "results" / "model_consistency.json")
    if rep:
        rows = []
        for label, v in rep["contrasts"].items():
            m = v["meta"]
            row = {"contrast": label}
            for model, cc in v["per_model"].items():
                row[model] = round(cc["estimate"], 3)
            if m.get("k", 0) >= 2:
                row.update({"pooled": round(m["random"], 3),
                            "95% CI": f"[{m['random_ci'][0]:.3f}, {m['random_ci'][1]:.3f}]",
                            "I²": f"{m['I2']:.0%}", "same direction": f"{m['same_sign']}/{m['k']}"})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("##### Does creativity pay off?")
    st.caption("Quality-diversity frontier (pass rate against novelty per condition) and, on Tier 3, whether "
               "more novel heuristics score better than the standard baseline.")
    if st.button("Compute frontier and Tier 3 payoff", key="frontier"):
        _start(st, jobs, "frontier", "frontier.py", ["--plot"])
    c["live"]("frontier", lambda: c["job_panel"]("frontier", "Frontier"))
    fr = _read(ROOT / "data" / "results" / "frontier.json")
    if fr:
        st.dataframe(pd.DataFrame([{"condition": k, **v} for k, v in fr["frontier"].items()]),
                     hide_index=True, use_container_width=True)
        if fr.get("tier3"):
            st.dataframe(pd.DataFrame([{"condition": k, **v} for k, v in fr["tier3"].items()]),
                         hide_index=True, use_container_width=True)
        png = ROOT / "data" / "results" / "frontier.png"
        if png.exists():
            st.image(str(png), width=520)

    st.markdown("##### Are the held-out tests strong enough? (mutation testing)")
    st.caption("Makes small changes to correct solutions and checks the held-out tests catch them, on five tasks. "
               "No model calls.")
    if st.button("Run mutation testing", key="mutation"):
        _start(st, jobs, "mutation", "mutation_test.py", ["--sandbox", c["sandbox"]])
    c["live"]("mutation", lambda: c["job_panel"]("mutation", "Mutation testing"))
    mu = _read(ROOT / "data" / "results" / "mutation.json")
    if mu:
        st.dataframe(pd.DataFrame([{"task": k, "solutions": v["solutions"], "mutants": v["mutants"],
                                    "killed": v["killed"], "mutation score": v["score"]} for k, v in mu.items()]),
                     hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- References: sensitivity

def reference_sensitivity(st, c) -> None:
    jobs = c["jobs"]
    st.markdown("##### Sensitivity to the reference size (50 / 100 / 200)")
    st.caption("First build 200 samples for a few tasks above (Samples per task = 200, Choose). This check then "
               "scores the solutions against the first 50, 100 and 200 and reports whether the novelty ranking "
               "and the 'novel' flag agree. The main analysis always uses the first 100.")
    if st.button("Run sensitivity check", key="refsens"):
        _start(st, jobs, "refsens", "reference_sensitivity.py", [])
    c["live"]("refsens", lambda: c["job_panel"]("refsens", "Reference sensitivity"))


# ---------------------------------------------------------------- Rating study (SC4)

def rating_study(st, c) -> None:
    pd, jobs, ROOT = c["pd"], c["jobs"], c["ROOT"]
    rdir = ROOT / "data" / "rating"
    st.subheader("Rating study (SC4): humans and the AI judge")
    st.caption("60 pairs of correct solutions to the same task. Raters choose which is more original and which "
               "is more creative; their combined ranking is compared with the automatic novelty score.")
    st.markdown("**1. Build the pairs** (from the current traces; frozen once responses are imported)")
    n = st.number_input("Pairs", 10, 200, 60, key="rs_n")
    if st.button("Build pairs", key="rs_build"):
        _start(st, jobs, "rating", "rating_study.py", ["build", "--n", str(int(n)), "--sandbox", c["sandbox"]])
    st.markdown("**2. Export the rater materials**: a booklet to print to PDF, and a Word file for Microsoft "
                "Forms → Quick import. The file already holds the information text, the consent question, Part A (pairs) and Part B (labelling).")
    if st.button("Export booklet and form", key="rs_export"):
        _start(st, jobs, "rating", "rating_study.py", ["export"])
    c["live"]("rating", lambda: c["job_panel"]("rating", "Rating study step"))
    for name, mime in (("rating_booklet.html", "text/html"),
                       ("form_quick_import.docx",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")):
        p = rdir / name
        if p.exists():
            st.download_button(f"Download {name}", p.read_bytes(), file_name=name, mime=mime, key=f"dl_{name}")
    st.markdown("**3. Import the responses** (the Form's Excel export)")
    up = st.file_uploader("Responses (.xlsx or .csv)", type=["xlsx", "csv"], key="rs_up")
    if up is not None and st.button("Import responses", key="rs_import"):
        rdir.mkdir(parents=True, exist_ok=True)
        dest = rdir / ("responses_upload" + Path(up.name).suffix)
        dest.write_bytes(up.getvalue())
        _start(st, jobs, "rating", "rating_study.py", ["import", str(dest)])
    st.markdown("**4. The independent AI judge** (profile `deepseek`; needs `ollama pull deepseek-coder-v2:16b`)")
    if st.button("Run the AI judge", key="rs_judge"):
        _start(st, jobs, "judge", "rating_study.py", ["judge", "--profile", "deepseek"])
    c["live"]("judge", lambda: c["job_panel"]("judge", "AI judge"))
    st.markdown("**5. Analyse**")
    if st.button("Analyse ratings", type="primary", key="rs_analyse"):
        _start(st, jobs, "rating", "rating_study.py", ["analyse"])
    res = _read(rdir / "results.json")
    if res:
        rows = []
        for who in ("humans", "judge"):
            if who not in res:
                continue
            for q, v in res[who]["questions"].items():
                rows.append({"raters": who, "question": q, "Spearman rho": v["spearman_rho"], "95% CI": v["rho_ci"],
                             "Cohen's kappa": v["mean_cohen_kappa"], "Fleiss' kappa": v["fleiss_kappa"],
                             "majority picks more novel": v["majority_picks_more_novel"], "pairs": v["n_pairs"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("SC4 targets: Spearman's rho of at least 0.50 against the automatic novelty; Cohen's kappa of "
                   "at least 0.60 between raters. The judge is reported separately, never pooled with humans.")
        if "judge_vs_humans" in res:
            st.write("AI judge agreement with the human majority:", res["judge_vs_humans"])


# ---------------------------------------------------------------- Open problems (search study)

def open_problems(st, c) -> None:
    pd, jobs, ROOT = c["pd"], c["jobs"], c["ROOT"]
    st.subheader("Open and very hard problems: search study")
    st.caption("Each strategy writes and scores the same number of programs. random = independent tries; "
               "single = one model improving the best programs so far; studio = Innovator + Engineer + "
               "Facilitator. Score 1.0 = the standard baseline heuristic; higher is better.")
    hard = [t for t in c["bank"] if t.scoring == "heuristic"]
    pick = st.multiselect("Problems", [t.id for t in hard], default=[t.id for t in hard if t.id in
                          ("t3_circle_packing", "t3_bin_packing", "t3_cap_set")], key="op_tasks")
    strategies = st.multiselect("Strategies", ["random", "single", "studio"], default=["random", "single", "studio"],
                                key="op_strats")
    c1, c2 = st.columns(2)
    budget = c1.number_input("Programs per run (budget)", 5, 500, 40, key="op_budget")
    seeds = c2.text_input("Seeds", "1-3", key="op_seeds")
    if st.button("Start search", type="primary", disabled=not (pick and strategies), key="op_run"):
        _start(st, jobs, "search", "search_open.py",
               ["run", "--tasks", ",".join(pick), "--strategies", ",".join(strategies), "--budget", str(int(budget)),
                "--seeds", seeds, "--profile", c["profile"], "--sandbox", c["sandbox"]])
    c["live"]("search", lambda: c["job_panel"]("search", "Search"))
    if st.button("Summarise results", key="op_report"):
        _start(st, jobs, "search_report", "search_open.py", ["report"])
    c["live"]("search_report", lambda: c["job_panel"]("search_report", "Search report"))
    rep = _read(ROOT / "data" / "results" / "search_report.json")
    if rep:
        models = list(rep.get("models", {})) or [None]
        mpick = st.selectbox("Model", models, index=models.index(rep["main_model"]) if rep.get("main_model") in models else 0,
                             key="op_model") if models != [None] else None
        tasks = rep["models"][mpick]["tasks"] if mpick else rep["tasks"]
        for tid, t in tasks.items():
            st.markdown(f"**{tid}**" + (f" · best known raw value {t['best_known']}" if t.get("best_known") else ""))
            st.dataframe(pd.DataFrame([{"strategy": s, "best score (mean over seeds)": round(v["best_mean"], 4),
                                        "best raw value": v["best_raw"], "gap to best known": v["gap_to_best_known"],
                                        "AUC": round(v["auc_mean"], 4), "valid share": round(v["valid_share"], 2),
                                        "tokens": int(v["tokens_mean"]), "interventions": v["interventions"]}
                                       for s, v in t["strategies"].items()]), hide_index=True, use_container_width=True)
            if t.get("comparisons"):
                st.write(t["comparisons"])
    # best-so-far curves
    from macs.search.loop import read_curve
    sroot = ROOT / "data" / "search"
    logs = (sorted(sroot.glob("*/*.jsonl")) + sorted(sroot.glob("*.jsonl"))) if sroot.exists() else []
    if logs and rep and rep.get("models") and mpick:
        logs = [p for p in logs if read_curve(p)["model"] == mpick]
    if logs:
        task_pick = st.selectbox("Curves for", sorted({p.name.split("-")[0] for p in logs}), key="op_curve")
        curves = {}
        for p in logs:
            if p.name.split("-")[0] != task_pick:
                continue
            r = read_curve(p)
            curves[f"{r['strategy']} s{r['seed']}"] = r["curve"]
        if curves:
            n = max(len(v) for v in curves.values())
            st.line_chart(pd.DataFrame({k: v + [None] * (n - len(v)) for k, v in curves.items()}))
            st.caption("Best score so far against the number of programs scored.")
