"""Turn traces into the study's numbers: per-session metrics, aggregated by condition.

For each session:
  * diversity   - how many solution families the proposals span (clustered AST features)
  * correctness - how many proposals pass the held-out tests (Tier 3: are feasible)
  * novelty     - how unusual the proposals are relative to the task's frozen reference set
                  (only where a reference exists)
  * the joint correct-and-novel rate, Tier 3 scores, tokens used and interventions.
These are averaged by condition (A / B / C / D / D_minus). pooled_diversity adds the fair
tutor-vs-studio comparison: families at a matched number of solutions per task.
"""
from __future__ import annotations

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


def _embed(codes: list[str]) -> tuple[np.ndarray, list[int]]:
    """AST-feature matrix for the codes that parse; returns (matrix, indices kept)."""
    rows, keep = [], []
    for i, c in enumerate(codes):
        try:
            rows.append(ast_features(normalise(c))); keep.append(i)
        except Exception:  # noqa: BLE001
            pass
    return (np.vstack(rows) if rows else np.empty((0, 0))), keep


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


def session_metrics(path, *, backend="subprocess", reference=None, cluster_threshold=CLUSTER_THRESHOLD) -> dict:
    events = list(read_trace(path))
    start = next(e for e in events if e["event"] == "session_start")
    task = task_by_id(start["task_id"])
    codes = [c for _, c in implementations(events)]

    # --- correctness / quality ---
    # Tier 1/2 are pass/fail on held-out tests ('correct' = passed all). Tier 3 are metric-scored
    # ('correct' = feasible), and the heuristic score relative to the baseline is kept as well
    # (>1 beats it). Both feed `correct_flags`, so the correct-and-novel rate works for either.
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
        Xc, scale_source = _standardise_for_clustering(X)
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
            "avg_tokens": avg("tokens"),
            "avg_interventions": avg("n_interventions"),
            "n_stopped_by_token_ceiling": sum(1 for r in rows if r.get("stopped_by_token_ceiling")),
        }
    return out


def pooled_diversity(codes_by_task_cond: dict[tuple[str, str], list[str]], *,
                     cluster_threshold=CLUSTER_THRESHOLD, seed: int = 0) -> dict:
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
        rows, owner = [], []
        for cond, codes in sorted(conds.items()):
            for c in codes:
                try:
                    rows.append(ast_features(normalise(c))); owner.append(cond)
                except Exception:  # noqa: BLE001 - unparseable code has no features
                    pass
        if not rows:
            continue
        X = np.vstack(rows)
        Xc = StandardScaler().fit_transform(X) if len(X) >= 2 else X
        labels = cluster_families(Xc, distance_threshold=cluster_threshold)
        owner = np.array(owner)
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


def analyse_dir(traces_dir="data/traces", *, backend="subprocess", references=None, return_pooled=False,
                include_legacy=False):
    """Score + aggregate every trace in a directory. `references` maps task_id -> ReferenceStats.

    Traces whose task_id is no longer in the task bank are skipped with a printed note. Traces from
    an older orchestrator version, and sessions that crashed before completing, are excluded unless
    include_legacy=True.
    """
    from macs.orchestrator import ORCHESTRATOR_VERSION
    refs = references or {}
    sessions, skipped, legacy, incomplete = [], [], [], []
    pooled_codes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in sorted(Path(traces_dir).glob("*.jsonl")):
        events = list(read_trace(p))
        start = next((e for e in events if e["event"] == "session_start"), None)
        if not start:
            continue
        # traces from an older orchestrator version are not comparable with current ones
        version = start.get("header", {}).get("orchestrator_version", "0.1.0")
        if not include_legacy and _version_tuple(version) < _version_tuple(ORCHESTRATOR_VERSION):
            legacy.append(p.name)
            continue
        end = events[-1] if events and events[-1].get("event") == "session_end" else {}
        if not include_legacy and not str(end.get("content", "")).startswith(("completed", "stopped")):
            incomplete.append(p.name)       # crashed mid-session (e.g. model server down)
            continue
        try:
            sessions.append(session_metrics(p, backend=backend, reference=refs.get(start["task_id"])))
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
    if legacy:
        print(f"[analyse] skipped {len(legacy)} trace(s) from an older orchestrator "
              f"(< {ORCHESTRATOR_VERSION}); use --include-old to keep them")
    summary = aggregate_by_condition(sessions)
    pooled = pooled_diversity(pooled_codes)
    for cond, v in pooled["by_condition"].items():
        if cond in summary:
            summary[cond]["avg_families_matched"] = v["avg_families_at_matched_m"]
    if return_pooled:
        return sessions, summary, pooled
    return sessions, summary
