"""Reference-size sensitivity check: does novelty change if the reference has 50 or 200
samples instead of 100? (Proposal, section 3.6: "sensitivity to 50 and 200 is checked".)

For every task that has a reference set and traces, every solution in the traces is scored
against the first 50, 100 and 200 reference samples, and per task the script reports:
  * the Spearman rank correlation of novelty between 100 and 50 / 200 (1.0 = same ranking),
  * how often the 'novel' flag (top decile) agrees between 100 and 50 / 200.
Sizes the reference file does not yet have are skipped. Build 200 samples on some tasks first:

    python scripts/build_reference.py --tasks t1_mbpp19_test_duplicate,t3_bin_packing --n 200
    python scripts/reference_sensitivity.py

Results are printed and saved to data/results/reference_sensitivity.json. The main analysis
always uses the first 100 samples (REFERENCE_N).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from macs.analysis.aggregate import NOVELTY_PERCENTILE
from macs.analysis.score import implementations
from macs.metrics.reference import REFERENCE_ROOT, REFERENCE_N, build_reference_stats, load_reference_solutions
from macs.metrics.represent import ast_features, normalise
from macs.trace.writer import read_trace

SIZES = (50, REFERENCE_N, 200)


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/traces")
    ap.add_argument("--out", default="data/results/reference_sensitivity.json")
    args = ap.parse_args()

    codes_by_task: dict[str, list[str]] = defaultdict(list)
    for p in sorted(Path(args.traces).glob("*.jsonl")):
        events = list(read_trace(p))
        start = next((e for e in events if e["event"] == "session_start"), None)
        if start:
            codes_by_task[start["task_id"]].extend(c for _, c in implementations(events))

    report = {}
    for task_id, codes in sorted(codes_by_task.items()):
        if not (Path(REFERENCE_ROOT) / f"{task_id}.jsonl").exists():
            continue
        available = len(load_reference_solutions(task_id, n=None))
        X = []
        for c in codes:
            try:
                X.append(ast_features(normalise(c)))
            except Exception:  # noqa: BLE001
                pass
        if not X:
            continue
        X = np.vstack(X)
        nov, flag = {}, {}
        for n in SIZES:
            if available < n:
                continue
            rs = build_reference_stats(task_id, n=n)
            nov[n] = rs.novelty(X)
            flag[n] = rs.percentile(X) >= NOVELTY_PERCENTILE
        row = {"reference_available": available, "n_solutions": int(len(X)), "sizes_checked": sorted(nov)}
        if REFERENCE_N in nov:
            for n in SIZES:
                if n != REFERENCE_N and n in nov:
                    row[f"spearman_{REFERENCE_N}_vs_{n}"] = _spearman(nov[REFERENCE_N], nov[n])
                    row[f"novel_flag_agreement_{REFERENCE_N}_vs_{n}"] = float((flag[REFERENCE_N] == flag[n]).mean())
        report[task_id] = row
        print(task_id, json.dumps(row))

    if not report:
        print("nothing to check: need traces plus a reference set with at least 50 samples")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
