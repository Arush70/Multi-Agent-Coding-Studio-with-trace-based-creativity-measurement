"""Turn traces into the study's numbers: per-session metrics, aggregated by condition.

For each session:
  * diversity   - how many solution families the proposals span (clustered AST features)
  * correctness - how many proposals pass the held-out tests (Tier 3: are feasible)
  * novelty     - how unusual the proposals are relative to the task's frozen reference set
                  (only where a reference exists)
  * the joint correct-and-novel rate, Tier 3 scores, tokens used and interventions.
These are averaged by condition (see macs/conditions.py). pooled_diversity adds the fair
tutor-vs-studio comparison: families at a matched number of solutions per task.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.preprocessing import StandardScaler

from macs.trace.writer import read_trace
from macs.metrics.represent import ast_features, normalise
from macs.metrics.diversity import cluster_families, n_families, family_entropy, rarefaction
from macs.metrics.usefulness import correct_and_novel_rate
from macs.analysis.score import implementations, task_by_id, score_code_heldout

# Euclidean distance in the standardised AST space, i.e. how many per-feature standard deviations
# apart two solutions must be to count as different families. 3.0 is provisional; the frozen value
# is set by SC3 calibration (scripts/calibrate_clustering.py: maximise ARI against blind hand
# labels) and recorded in docs/PREREGISTRATION.md before the main runs.
# Diversity is measured among the solutions themselves; novelty is measured against the frozen
# reference. They deliberately use different spaces, so a homogeneous reference cannot collapse the
# diversity measure.
CLUSTER_THRESHOLD = 3.0
NOVELTY_PERCENTILE = 0.90
# Parallel representation (proposal: AST and UniXcoder analysed separately, never combined). In
# UniXcoder space families are clustered by cosine distance (embeddings are unit-norm, so no
# standardisation), average linkage. Provisional; frozen by SC3 calibration in that space
# (scripts/calibrate_clustering.py --representation unixcoder).
UNIXCODER_CLUSTER_THRESHOLD = 0.10
REPRESENTATIONS = ("ast", "unixcoder")


def _embed(codes: list[str], representation: str = "ast") -> tuple[np.ndarray, list[int]]:
    """Feature matrix for the codes that parse, in the chosen representation; (matrix, indices kept)."""
    normed, keep = [], []
    for i, c in enumerate(codes):
        try:
            n = normalise(c)
            if representation == "ast":
                normed.append(ast_features(n))
            else:
                normed.append(n)
            keep.append(i)
        except Exception:  # noqa: BLE001
            pass
    if not normed:
        return np.empty((0, 0)), keep
    if representation == "ast":
        return np.vstack(normed), keep
    from macs.metrics.represent import embed
    return embed(normed), keep


def _cluster(X: np.ndarray, representation: str, threshold: float | None = None) -> tuple[np.ndarray, str]:
    """Family labels in the representation's own space and rule; returns (labels, scale note)."""
    if representation == "ast":
        Xc, note = _standardise_for_clustering(X)
        return cluster_families(Xc, distance_threshold=CLUSTER_THRESHOLD if threshold is None else threshold), note
    return cluster_families(X, distance_threshold=UNIXCODER_CLUSTER_THRESHOLD if threshold is None else threshold,
                            metric="cosine"), "unit-norm"


def _standardise_for_clustering(X: np.ndarray) -> tuple[np.ndarray, str]:
    """Put AST features on a common scale BEFORE clustering, so raw counts (n_lines, n_call...)
    don't dominate and collapse every strategy into one family.

    Standardisation is within the session's own proposals, not against the reference: diversity is
    about spread among the proposals, and a homogeneous reference set would flatten the scale and
    collapse distinct strategies into one family. Novelty, a distance to the reference, still uses
    the reference.
    """
    if len(X) < 2:
        return X, "single"  # one proposal -> single family anyway
    return StandardScaler().fit_transform(X), "within-sample"


def session_metrics(path, *, backend="subprocess", reference=None, cluster_threshold=None,
                    representation: str = "ast") -> dict:
    events = list(read_trace(path))
    start = next(e for e in events if e["event"] == "session_start")
    task = task_by_id(start["task_id"])
    codes = [c for _, c in implementations(events)]

    # --- correctness / quality ---
    # Tier 1/2 are pass/fail on held-out tests ('correct' = passed all). Tier 3 are metric-scored
    # ('correct' = feasible), and the heuristic score relative to the baseline is kept as well
    # (>1 beats it). Both feed `correct_flags`, so the correct-and-novel rate works for either.
    correct_flags = []
    heur_scores: list[float | None] = []
    heur_mean = heur_best = None
    if task.scoring == "tests":
        for c in codes:
            correct_flags.append(bool(score_code_heldout(task, c, backend=backend).get("passed_all")))
    elif task.scoring == "heuristic":
        from macs.analysis.heuristic_score import score_heuristic_code
        scores = []
        for c in codes:
            r = score_heuristic_code(task, c, backend=backend)
            correct_flags.append(bool(r["valid"]))
            heur_scores.append(r["score"])
            if r["valid"] and r["score"] is not None:
                scores.append(r["score"])
        if scores:
            heur_mean = float(np.mean(scores)); heur_best = float(max(scores))
    n_correct = int(sum(correct_flags))

    # --- diversity (families among proposals) ---
    X, keep = _embed(codes, representation)
    scale_source = "none"
    if len(X) >= 1:
        labels, scale_source = _cluster(X, representation, cluster_threshold)
        fams = n_families(labels); ent = family_entropy(labels)
        fams_first10 = n_families(labels[:10])
    else:
        fams = ent = fams_first10 = 0

    # --- novelty (vs frozen reference, if available) ---
    mean_novelty = None; cn_rate = None
    z = None
    if reference is not None and len(X) >= 1:
        z = reference.novelty(X)
        mean_novelty = float(np.mean(z))
        pct = reference.percentile(X)
        is_novel = pct >= NOVELTY_PERCENTILE
        # align correctness to the kept (parseable) rows
        corr_kept = np.array([correct_flags[i] for i in keep]) if correct_flags else np.zeros(len(keep), bool)
        cn_rate = correct_and_novel_rate(corr_kept, is_novel)

    # per-proposal record (for the quality-diversity frontier and "does creativity pay off")
    zmap = {i: float(z[j]) for j, i in enumerate(keep)} if z is not None else {}
    proposals = [{"correct": bool(correct_flags[i]) if i < len(correct_flags) else None,
                  "novelty": zmap.get(i),
                  "heuristic_score": heur_scores[i] if i < len(heur_scores) else None}
                 for i in range(len(codes))]

    # --- learner contribution (RQ3, exploratory) ---
    from macs.metrics.learners import learner_contribution
    lc = learner_contribution(events, _reference_codes(task.id, start.get("header", {}).get("model")))

    # --- cost (proposal: token counts reported with every contrast) ---
    tokens = sum(int(e.get("prompt_tokens", 0) or 0) + int(e.get("completion_tokens", 0) or 0) for e in events)
    end = next((e for e in events if e["event"] == "session_end"), {})
    header = start.get("header", {})

    return {
        "session_id": start["session_id"], "task_id": task.id, "condition": start["condition"],
        "seed": header.get("seed"), "orchestrator_version": header.get("orchestrator_version", "0.1.0"),
        "tokens": tokens, "stopped_by_token_ceiling": str(end.get("content", "")).startswith("stopped"),
        "n_interventions": sum(1 for e in events if e["event"] == "intervention"),
        "n_revisions": sum(1 for e in events if e["event"] == "revision"),
        "n_proposals": len(codes), "n_correct": n_correct,
        "families": fams, "families_first10": fams_first10, "entropy": ent,
        "diversity_scale": scale_source,  # 'within-sample', or 'single' for one proposal
        "mean_novelty": mean_novelty, "correct_and_novel_rate": cn_rate,
        "scoring": task.scoring,          # 'tests' or 'heuristic'
        "heuristic_score": heur_mean,     # Tier 3 only: mean score vs baseline (>1 beats it)
        "heuristic_best": heur_best,      # Tier 3 only: best proposal's score vs baseline
        "learner_uptake_final": lc["uptake_final"], "learner_uptake_any": lc["uptake_any"],
        "learner_led_share": lc["learner_led_share"], "final_from_learners": lc["final_from_learners"],
        "learner_novel_share": lc["learner_novel_share"], "learner_constructs": lc["learner_constructs"],
        "proposals": proposals,
    }


_REF_CODES: dict[tuple[str, str | None], list[str]] = {}


def _reference_codes(task_id: str, model: str | None) -> list[str]:
    """The task's reference solutions for this model (empty if none), loaded once per process."""
    key = (task_id, model)
    if key not in _REF_CODES:
        from macs.metrics.reference import load_reference_solutions
        try:
            _REF_CODES[key] = load_reference_solutions(task_id, model=model)
        except (FileNotFoundError, OSError, ValueError):
            _REF_CODES[key] = []
    return _REF_CODES[key]


def aggregate_by_condition(session_dicts: list[dict]) -> dict:
    """Average each metric within each condition -> the headline comparison table."""
    by = defaultdict(list)
    for s in session_dicts:
        by[s["condition"]].append(s)
    out = {}
    for cond, rows in sorted(by.items()):
        def avg(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(mean(vals), 3) if vals else None
        out[cond] = {
            "n_sessions": len(rows),
            "avg_families": avg("families"),
            "avg_families_first10": avg("families_first10"),
            "avg_entropy": avg("entropy"),
            "avg_correct": avg("n_correct"),
            "avg_mean_novelty": avg("mean_novelty"),
            "avg_correct_and_novel": avg("correct_and_novel_rate"),
            "avg_heuristic_score": avg("heuristic_score"),   # None for pure test conditions
            "avg_heuristic_best": avg("heuristic_best"),
            "avg_tokens": avg("tokens"),
            "avg_interventions": avg("n_interventions"),
            "n_stopped_by_token_ceiling": sum(1 for r in rows if r.get("stopped_by_token_ceiling")),
            # RQ3 (exploratory): how much of the solver's output traces back to the learners
            "avg_learner_uptake_final": avg("learner_uptake_final"),
            "avg_learner_led_share": avg("learner_led_share"),
            "avg_learner_novel_share": avg("learner_novel_share"),
        }
    return out


def pooled_diversity(codes_by_task_cond: dict[tuple[str, str], list[str]], *,
                     cluster_threshold=None, seed: int = 0, representation: str = "ast") -> dict:
    """Families per condition at a MATCHED number of solutions, per task.

    The per-session family count cannot compare tutor and studio: a tutor session writes one
    program (so it is always one family) while a studio session writes several. Here, for each
    task, all solutions from all conditions and repetitions are put in one common standardised AST
    space, clustered once, and each condition's family count is read off its rarefaction curve at
    the same number of solutions m (the smallest condition's count for that task). Repetitions
    (seeds) give the tutor conditions more than one solution.
    """
    by_task: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for (task_id, cond), codes in codes_by_task_cond.items():
        by_task[task_id][cond] = codes
    rng = np.random.default_rng(seed)
    per_task: dict[str, dict] = {}
    for task_id, conds in sorted(by_task.items()):
        all_codes, owner = [], []
        for cond, codes in sorted(conds.items()):
            for c in codes:
                all_codes.append(c); owner.append(cond)
        X, keep = _embed(all_codes, representation)
        if not keep:
            continue
        owner = np.array([owner[i] for i in keep])
        if representation == "ast":
            Xc = StandardScaler().fit_transform(X) if len(X) >= 2 else X
            labels = cluster_families(Xc, distance_threshold=CLUSTER_THRESHOLD if cluster_threshold is None
                                      else cluster_threshold)
        else:
            labels, _ = _cluster(X, representation, cluster_threshold)
        counts = {c: int((owner == c).sum()) for c in sorted(set(owner))}
        m = min(counts.values())
        out = {"m": m, "n_by_condition": counts, "families_at_m": {}, "families_total": {}}
        for c in counts:
            lab = labels[owner == c]
            curve = rarefaction(lab, rng)
            out["families_at_m"][c] = round(float(curve[m - 1]), 3)
            out["families_total"][c] = n_families(lab)
        per_task[task_id] = out
    by_cond = defaultdict(list)
    for t in per_task.values():
        for c, v in t["families_at_m"].items():
            by_cond[c].append(v)
    summary = {c: {"n_tasks": len(v), "avg_families_at_matched_m": round(mean(v), 3)} for c, v in sorted(by_cond.items())}
    return {"per_task": per_task, "by_condition": summary}


def _version_tuple(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (0,)


def _model_id(start: dict) -> str:
    h = start.get("header", {})
    return f"{h.get('model', '?')}@{h.get('revision', '')}"


def trace_models(traces_dir="data/traces") -> dict[str, str]:
    """{model@revision: timestamp of its newest trace} for every trace in a directory."""
    out: dict[str, str] = {}
    for p in Path(traces_dir).glob("*.jsonl"):
        try:
            with open(p, encoding="utf-8") as fh:
                first = json.loads(fh.readline())
        except (OSError, ValueError):
            continue
        if first.get("event") != "session_start":
            continue
        mid = _model_id(first)
        out[mid] = max(out.get(mid, ""), str(first.get("ts", "")))
    return out


def resolve_model(traces_dir="data/traces", model: str | None = None) -> str | None:
    """The model NAME an analysis will use: `model` (name or name@revision) if given, otherwise the
    model of the most recent trace. None when there are no traces."""
    if model:
        return model.split("@", 1)[0]
    models = trace_models(traces_dir)
    if not models:
        return None
    return max(models.items(), key=lambda kv: kv[1])[0].split("@", 1)[0]


def analyse_dir(traces_dir="data/traces", *, backend="subprocess", references=None, return_pooled=False,
                include_legacy=False, model=None, representation: str = "ast"):
    """Score + aggregate every trace in a directory. `references` maps task_id -> ReferenceStats.

    Traces whose task_id is no longer in the task bank are skipped with a printed note. Traces from
    an older orchestrator version, and sessions that crashed before completing, are excluded unless
    include_legacy=True.

    Results from different models are never pooled. `model` selects one (a model name, or
    "name@revision"); when it is None and the directory holds traces from several models, only the
    model of the most recent trace is analysed and the others are reported as skipped.
    """
    from macs.orchestrator import COMPATIBLE_SINCE
    refs = references or {}
    sessions, skipped, legacy, incomplete = [], [], [], []
    pooled_codes: dict[tuple[str, str], list[str]] = defaultdict(list)
    loaded = []
    for p in sorted(Path(traces_dir).glob("*.jsonl")):
        events = list(read_trace(p))
        start = next((e for e in events if e["event"] == "session_start"), None)
        if start:
            loaded.append((p, events, start))
    models = {_model_id(st) for _, _, st in loaded}
    if model is not None:
        target = {m for m in models if m == model or m.split("@", 1)[0] == model}
    elif len(models) > 1:
        newest = max(loaded, key=lambda x: str(x[2].get("ts", "")))[2]
        target = {_model_id(newest)}
    else:
        target = models
    other_model = [p.name for p, _, st in loaded if _model_id(st) not in target]
    for p, events, start in loaded:
        if _model_id(start) not in target:
            continue
        # traces from an older orchestrator version are not comparable with current ones
        version = start.get("header", {}).get("orchestrator_version", "0.1.0")
        if not include_legacy and _version_tuple(version) < _version_tuple(COMPATIBLE_SINCE):
            legacy.append(p.name)
            continue
        end = events[-1] if events and events[-1].get("event") == "session_end" else {}
        if not include_legacy and not str(end.get("content", "")).startswith(("completed", "stopped")):
            incomplete.append(p.name)       # crashed mid-session (e.g. model server down)
            continue
        try:
            sessions.append(session_metrics(p, backend=backend, reference=refs.get(start["task_id"]),
                                            representation=representation))
        except KeyError:
            skipped.append((p.name, start.get("task_id")))
            continue
        pooled_codes[(start["task_id"], start["condition"])].extend(c for _, c in implementations(events))
    if skipped:
        print(f"[analyse] skipped {len(skipped)} stale trace(s) for unknown tasks: "
              + ", ".join(sorted({t for _, t in skipped})))
    if incomplete:
        print(f"[analyse] skipped {len(incomplete)} incomplete trace(s) (session crashed): "
              + ", ".join(incomplete[:5]) + (" ..." if len(incomplete) > 5 else ""))
    if other_model:
        print(f"[analyse] analysing model {', '.join(sorted(target)) or model}; skipped "
              f"{len(other_model)} trace(s) from other models (results from different models are "
              f"never pooled; choose one with --model)")
    if legacy:
        print(f"[analyse] skipped {len(legacy)} trace(s) from an older orchestrator "
              f"(< {COMPATIBLE_SINCE}); use --include-old to keep them")
    summary = aggregate_by_condition(sessions)
    pooled = pooled_diversity(pooled_codes, representation=representation)
    for cond, v in pooled["by_condition"].items():
        if cond in summary:
            summary[cond]["avg_families_matched"] = v["avg_families_at_matched_m"]
    if return_pooled:
        return sessions, summary, pooled
    return sessions, summary
