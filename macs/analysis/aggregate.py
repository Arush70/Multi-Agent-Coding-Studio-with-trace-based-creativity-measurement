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

from macs.trace.writer import read_trace
from macs.metrics.represent import ast_features, normalise
from macs.metrics.diversity import cluster_families, n_families, family_entropy
from macs.metrics.usefulness import correct_and_novel_rate
from macs.analysis.score import implementations, task_by_id, score_code_heldout

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


def session_metrics(path, *, backend="subprocess", reference=None, cluster_threshold=CLUSTER_THRESHOLD) -> dict:
    events = list(read_trace(path))
    start = next(e for e in events if e["event"] == "session_start")
    task = task_by_id(start["task_id"])
    codes = [c for _, c in implementations(events)]

    # --- correctness (held-out) ---
    correct_flags = []
    if task.scoring == "tests":
        for c in codes:
            correct_flags.append(bool(score_code_heldout(task, c, backend=backend).get("passed_all")))
    n_correct = int(sum(correct_flags))

    # --- diversity (families among proposals) ---
    X, keep = _embed(codes)
    if len(X) >= 1:
        labels = cluster_families(X, distance_threshold=cluster_threshold)
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
        "mean_novelty": mean_novelty, "correct_and_novel_rate": cn_rate,
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
        }
    return out


def analyse_dir(traces_dir="data/traces", *, backend="subprocess", references=None) -> tuple[list[dict], dict]:
    """Score + aggregate every trace in a directory. `references` maps task_id -> ReferenceStats."""
    refs = references or {}
    sessions = []
    for p in sorted(Path(traces_dir).glob("*.jsonl")):
        start = next((e for e in read_trace(p) if e["event"] == "session_start"), None)
        if not start:
            continue
        ref = refs.get(start["task_id"])
        sessions.append(session_metrics(p, backend=backend, reference=ref))
    return sessions, aggregate_by_condition(sessions)
