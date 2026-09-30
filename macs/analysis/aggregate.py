"""Turn traces into the study's numbers: per-session metrics, aggregated by condition.

Arush - this is where a pile of trace files becomes your results table. For each session it
computes the three things your project measures:
  * diversity  - how many solution families the proposals span (clustered AST features)
  * correctness- how many proposals pass the held-out tests, and did the final one
  * novelty    - how unusual the proposals are vs the task's frozen reference set (only if that
                 reference exists; skipped otherwise, so this still runs on a pilot)
and the joint 'correct-and-novel' rate. Then it averages by condition so you can compare
A / B / C / D / D_minus directly - which is the whole experiment.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.preprocessing import StandardScaler

from macs.trace.writer import read_trace
from macs.metrics.represent import ast_features, normalise
from macs.metrics.diversity import cluster_families, n_families, family_entropy
from macs.metrics.usefulness import correct_and_novel_rate
from macs.analysis.score import implementations, task_by_id, score_code_heldout

# Arush - this threshold now applies in the STANDARDISED AST space (see _standardise_for_clustering),
# not on raw counts. 0.30 is provisional: it gives a sensible family count on the pilot. The frozen
# value is set by the SC3 calibration (scripts/calibrate_clustering.py -> maximise ARI vs your blind
# known_families) and recorded in docs/PREREGISTRATION.md before the main runs.
CLUSTER_THRESHOLD = 0.30
NOVELTY_PERCENTILE = 0.90


def _embed(codes: list[str]) -> tuple[np.ndarray, list[int]]:
    """AST-feature matrix for the codes that parse; returns (matrix, indices kept)."""
    rows, keep = [], []
    for i, c in enumerate(codes):
        try:
            rows.append(ast_features(normalise(c))); keep.append(i)
        except Exception:  # noqa: BLE001
            pass
    return (np.vstack(rows) if rows else np.empty((0, 0))), keep


def _standardise_for_clustering(X: np.ndarray, reference) -> tuple[np.ndarray, str]:
    """Put AST features on a common scale BEFORE clustering, so raw counts (n_lines, n_call...)
    don't dominate the cosine direction and collapse every strategy into one family.

    Arush - this is the fix for the 'families=1' pilot result. Clustering must live in the SAME
    space as novelty. If the task has a frozen reference set we reuse ITS scaler (the pre-registered
    path, byte-for-byte the scaling novelty applies); on a pilot with no reference yet we scale the
    proposals against themselves and flag it, so the number still means something but you know it's
    not the frozen path. Returns (X_scaled, scale_source).
    """
    if reference is not None and getattr(reference, "scaler", None) is not None:
        return reference.scaler.transform(X), "reference"
    if len(X) < 2:
        return X, "none"  # one proposal -> nothing to scale, single family anyway
    return StandardScaler().fit_transform(X), "within-sample"


def session_metrics(path, *, backend="subprocess", reference=None, cluster_threshold=CLUSTER_THRESHOLD) -> dict:
    events = list(read_trace(path))
    start = next(e for e in events if e["event"] == "session_start")
    task = task_by_id(start["task_id"])
    codes = [c for _, c in implementations(events)]

    # --- correctness / quality ---
    # Arush - Tier 1/2 are pass/fail on held-out tests ('correct' = passed all). Tier 3 are
    # metric-scored ('correct' = feasible), and we ALSO keep the heuristic score vs the baseline
    # (>1 beats it). Both feed the same `correct_flags`, so novelty's correct-and-novel rate works
    # for either kind.
    correct_flags = []
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
            if r["valid"] and r["score"] is not None:
                scores.append(r["score"])
        if scores:
            heur_mean = float(np.mean(scores)); heur_best = float(max(scores))
    n_correct = int(sum(correct_flags))

    # --- diversity (families among proposals) ---
    X, keep = _embed(codes)
    scale_source = "none"
    if len(X) >= 1:
        Xc, scale_source = _standardise_for_clustering(X, reference)
        labels = cluster_families(Xc, distance_threshold=cluster_threshold)
        fams = n_families(labels); ent = family_entropy(labels)
        fams_first10 = n_families(labels[:10])
    else:
        fams = ent = fams_first10 = 0

    # --- novelty (vs frozen reference, if available) ---
    mean_novelty = None; cn_rate = None
    if reference is not None and len(X) >= 1:
        z = reference.novelty(X)
        mean_novelty = float(np.mean(z))
        pct = reference.percentile(X)
        is_novel = pct >= NOVELTY_PERCENTILE
        # align correctness to the kept (parseable) rows
        corr_kept = np.array([correct_flags[i] for i in keep]) if correct_flags else np.zeros(len(keep), bool)
        cn_rate = correct_and_novel_rate(corr_kept, is_novel)

    return {
        "session_id": start["session_id"], "task_id": task.id, "condition": start["condition"],
        "n_proposals": len(codes), "n_correct": n_correct,
        "families": fams, "families_first10": fams_first10, "entropy": ent,
        "diversity_scale": scale_source,  # 'reference' = frozen path; 'within-sample' = pilot fallback
        "mean_novelty": mean_novelty, "correct_and_novel_rate": cn_rate,
        "scoring": task.scoring,          # 'tests' or 'heuristic'
        "heuristic_score": heur_mean,     # Tier 3 only: mean score vs baseline (>1 beats it)
        "heuristic_best": heur_best,      # Tier 3 only: best proposal's score vs baseline
    }


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
        }
    return out


def analyse_dir(traces_dir="data/traces", *, backend="subprocess", references=None) -> tuple[list[dict], dict]:
    """Score + aggregate every trace in a directory. `references` maps task_id -> ReferenceStats.

    Arush - traces whose task_id is no longer in the current bank (old pilots after you re-sourced
    the tasks) are skipped with a printed note, not crashed on, so one stale file can't kill the run.
    """
    refs = references or {}
    sessions, skipped = [], []
    for p in sorted(Path(traces_dir).glob("*.jsonl")):
        start = next((e for e in read_trace(p) if e["event"] == "session_start"), None)
        if not start:
            continue
        try:
            sessions.append(session_metrics(p, backend=backend, reference=refs.get(start["task_id"])))
        except KeyError:
            skipped.append((p.name, start.get("task_id")))
    if skipped:
        print(f"[analyse] skipped {len(skipped)} stale trace(s) for unknown tasks: "
              + ", ".join(sorted({t for _, t in skipped})))
    return sessions, aggregate_by_condition(sessions)
