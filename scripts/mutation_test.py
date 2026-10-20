"""Mutation testing of held-out test suites (proposal 3.7).

For each chosen task, up to three CORRECT solutions are taken from the task's reference set (the
model's ordinary solutions; one that passes every held-out test), mutated one change at a time,
and the held-out tests are run on every mutant. Reports the mutation score (share of mutants the
tests catch) per task. A low score means the held-out tests are too weak to tell a correct
solution from a subtly wrong one.

    python scripts/mutation_test.py                       # five default tasks
    python scripts/mutation_test.py --tasks t1_mbpp19_test_duplicate,t2_humaneval0_has_close_elements
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macs.analysis.mutation import mutation_score  # noqa: E402
from macs.analysis.score import score_code_heldout, task_by_id  # noqa: E402
from macs.metrics.reference import load_reference_solutions  # noqa: E402

DEFAULT = ("t1_mbpp19_test_duplicate,t1_mbpp140_extract_singly,t1_mbpp245_max_sum,"
           "t2_humaneval0_has_close_elements,t2_humaneval40_triples_sum_to_zero")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=DEFAULT)
    ap.add_argument("--per-task", type=int, default=3, help="correct solutions mutated per task")
    ap.add_argument("--model", default=None, help="whose reference set to take solutions from")
    ap.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"])
    ap.add_argument("--out", default="data/results/mutation.json")
    a = ap.parse_args()
    report = {}
    for tid in [t.strip() for t in a.tasks.split(",") if t.strip()]:
        task = task_by_id(tid)
        sols, seen = [], set()
        for code in load_reference_solutions(tid, model=a.model, n=None):
            if code in seen:
                continue
            seen.add(code)
            if score_code_heldout(task, code, backend=a.sandbox).get("passed_all"):
                sols.append(code)
            if len(sols) >= a.per_task:
                break
        runs = [mutation_score(task, c, backend=a.sandbox) for c in sols]
        total = sum(r["total"] for r in runs)
        killed = sum(r.get("killed", 0) for r in runs)
        report[tid] = {"solutions": len(sols), "mutants": total, "killed": killed,
                       "score": killed / total if total else None, "runs": runs}
        sc = "-" if not total else f"{killed / total:.2f}"
        print(f"{tid:<40} solutions={len(sols)} mutants={total:<4} killed={killed:<4} mutation score={sc}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
