"""SC3 calibration: freeze the clustering threshold by maximising ARI vs blind hand labels.

Arush - this is how you turn CLUSTER_THRESHOLD from a provisional guess into a defensible,
pre-registered number. You (or a second rater) hand-label a pool of solutions with which family
each belongs to - BLIND, i.e. without looking at what the automatic clustering says - then this
script sweeps the threshold and reports the one whose automatic families best match your labels,
scored by the Adjusted Rand Index (ARI). SC3's target is ARI >= 0.60.

Two steps, run once:

  1) build a labelling stub from your existing traces (blank 'family' field to fill in):
        python scripts/calibrate_clustering.py --make-stub
     -> writes data/calibration/labels.jsonl, one line per proposal. Open it and set each
        "family" to a short strategy name (reuse your task.yaml known_families vocabulary;
        use "other" for anything genuinely new). Do this without reading the analyze output.

  2) run the sweep once the labels are filled in:
        python scripts/calibrate_clustering.py
     -> prints ARI at each threshold (pooled over tasks) and the recommended value. Copy that
        into macs/analysis/aggregate.py (CLUSTER_THRESHOLD) and record it, with this table and
        the date, in docs/PREREGISTRATION.md. After that, never tune it against results again.

Clustering is per-task (families are task-specific) and runs in the SAME standardised AST space
the analysis uses: the frozen reference scaler when data/reference/<task_id>.jsonl exists, else
within-sample scaling (and a note, because the frozen path is the one you report).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from macs.trace.writer import read_trace
from macs.analysis.score import implementations
from macs.metrics.represent import ast_features, normalise
from macs.metrics.diversity import cluster_families, validate_against_labels

CALIB_PATH = Path("data/calibration/labels.jsonl")
REFERENCE_ROOT = Path("data/reference")
# the grid of thresholds to try, in the standardised AST space (cosine distance)
THRESHOLD_GRID = [round(t, 3) for t in np.arange(0.05, 0.81, 0.05)]


def _make_stub(traces_dir: str) -> None:
    """Pull every proposal out of the traces into a labelling file with blank family fields."""
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CALIB_PATH.exists():
        raise SystemExit(f"{CALIB_PATH} already exists - delete it first if you really want to rebuild it")
    n = 0
    with CALIB_PATH.open("w", encoding="utf-8") as out:
        for p in sorted(Path(traces_dir).glob("*.jsonl")):
            events = list(read_trace(p))
            start = next((e for e in events if e["event"] == "session_start"), None)
            if not start:
                continue
            for pid, code in implementations(events):
                out.write(json.dumps({
                    "task_id": start["task_id"], "session": p.stem, "proposal_id": pid,
                    "family": "",          # <-- YOU fill this in, blind
                    "code": code,
                }) + "\n")
                n += 1
    print(f"wrote {n} proposals to {CALIB_PATH}")
    print("Now open it and set every \"family\" to a short strategy name, then re-run without --make-stub.")


def _load_labels() -> dict[str, list[tuple[str, str]]]:
    """Return {task_id: [(code, family), ...]} for rows that actually have a family filled in."""
    if not CALIB_PATH.exists():
        raise SystemExit(f"no {CALIB_PATH}; run `python scripts/calibrate_clustering.py --make-stub` first")
    by_task: dict[str, list[tuple[str, str]]] = defaultdict(list)
    blank = 0
    for line in CALIB_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        fam = (rec.get("family") or "").strip()
        if not fam:
            blank += 1
            continue
        by_task[rec["task_id"]].append((rec["code"], fam))
    if blank:
        print(f"[warn] {blank} rows still have a blank family and were ignored.")
    return by_task


def _scaler_for(task_id: str):
    """Frozen reference scaler if a reference set exists for this task, else None (within-sample)."""
    ref = REFERENCE_ROOT / f"{task_id}.jsonl"
    if not ref.exists():
        return None
    from macs.metrics.reference import build_reference_stats
    return build_reference_stats(task_id).scaler  # AST reference is standardised by default


def _feature_matrix(codes: list[str]) -> tuple[np.ndarray, list[int]]:
    rows, keep = [], []
    for i, c in enumerate(codes):
        try:
            rows.append(ast_features(normalise(c))); keep.append(i)
        except Exception:  # noqa: BLE001 - a sample that won't parse is dropped, not guessed
            pass
    return (np.vstack(rows) if rows else np.empty((0, 0))), keep


def calibrate(by_task: dict[str, list[tuple[str, str]]]) -> None:
    # per (task, threshold) ARI, then averaged over tasks so no single big task dominates
    per_threshold: dict[float, list[float]] = defaultdict(list)
    print(f"\ncalibrating on {sum(len(v) for v in by_task.values())} labelled proposals "
          f"across {len(by_task)} task(s)\n")
    for task_id, rows in sorted(by_task.items()):
        codes = [c for c, _ in rows]
        hand = [f for _, f in rows]
        X, keep = _feature_matrix(codes)
        if len(X) < 2:
            print(f"  {task_id}: <2 parseable labelled solutions, skipped")
            continue
        hand = [hand[i] for i in keep]
        scaler = _scaler_for(task_id)
        Xs = scaler.transform(X) if scaler is not None else StandardScaler().fit_transform(X)
        src = "reference" if scaler is not None else "within-sample"
        best = (-2.0, None)
        for thr in THRESHOLD_GRID:
            labels = cluster_families(Xs, distance_threshold=thr)
            ari = validate_against_labels(labels, hand)
            per_threshold[thr].append(ari)
            if ari > best[0]:
                best = (ari, thr)
        print(f"  {task_id:<32} scale={src:<13} best ARI={best[0]:.3f} @ thr={best[1]}")

    print("\n threshold   mean ARI over tasks")
    print(" ---------   -------------------")
    pooled = {thr: float(np.mean(v)) for thr, v in per_threshold.items() if v}
    for thr in THRESHOLD_GRID:
        if thr in pooled:
            star = "  <-- best" if thr == max(pooled, key=pooled.get) else ""
            print(f"   {thr:<9} {pooled[thr]:+.3f}{star}")
    if pooled:
        best_thr = max(pooled, key=pooled.get)
        best_ari = pooled[best_thr]
        print(f"\nRECOMMENDED CLUSTER_THRESHOLD = {best_thr}  (mean ARI {best_ari:.3f}, "
              f"SC3 target >= 0.60 {'MET' if best_ari >= 0.60 else 'NOT met - inspect labels/features'})")
        print("Record this value + the table + today's date in docs/PREREGISTRATION.md, then freeze it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-stub", action="store_true", help="build data/calibration/labels.jsonl from traces")
    ap.add_argument("--traces", default="data/traces", help="traces directory for --make-stub")
    args = ap.parse_args()
    if args.make_stub:
        _make_stub(args.traces)
        return
    calibrate(_load_labels())


if __name__ == "__main__":
    main()
