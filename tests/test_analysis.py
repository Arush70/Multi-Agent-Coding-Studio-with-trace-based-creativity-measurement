"""Analysis-pipeline tests: held-out scoring + aggregation, with NO model.

Arush - writes trace files by hand with known solutions, then checks the scorer marks the
correct one correct and the wrong one wrong, and that the per-condition table adds up. This
tests the part that produces your actual results.
"""
from __future__ import annotations
import sys
import pytest

from macs.trace.writer import TraceWriter
from macs.analysis.score import score_trace
from macs.analysis.aggregate import session_metrics, aggregate_by_condition

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="subprocess sandbox rlimits differ on Windows; CI is Linux")

TASK_ID = "t1_mbpp19_test_duplicate"
CORRECT = "def test_duplicate(a):\n    return len(a) != len(set(a))\n"
WRONG = "def test_duplicate(a):\n    return False\n"
CORRECT2 = "def test_duplicate(a):\n    seen=set()\n    for x in a:\n        if x in seen: return True\n        seen.add(x)\n    return False\n"

HEADER = {"model": "fake", "revision": "0", "quantisation": "none", "temperature": 0.7,
          "seed": 1, "prompt_hashes": {"x": "y"}}


def _write(path, condition, impls):
    with TraceWriter(path, session_id=path.stem, task_id=TASK_ID, condition=condition, header=HEADER) as tr:
        for i, code in enumerate(impls):
            tr.log("engineer", "implementation", content=code, proposal_id=f"p{i}")


@posix_only
def test_scorer_marks_correct_and_wrong(tmp_path):
    p = tmp_path / "s1.jsonl"
    _write(p, "B", [CORRECT, WRONG])           # final is WRONG
    r = score_trace(p, backend="subprocess")
    assert r["n_proposals"] == 2
    assert r["n_correct_proposals"] == 1       # only CORRECT passes held-out
    assert r["final_passed_heldout"] is False  # final solution was WRONG


@posix_only
def test_session_metrics_diversity_and_correctness(tmp_path):
    p = tmp_path / "s2.jsonl"
    _write(p, "B", [CORRECT, CORRECT2, WRONG])
    m = session_metrics(p, backend="subprocess")
    assert m["n_proposals"] == 3
    assert m["n_correct"] == 2                  # two correct solutions
    assert m["families"] >= 1                   # families computed (default threshold is provisional)
    # at a tight threshold the three distinct strategies DO separate - proves clustering works
    tight = session_metrics(p, backend="subprocess", cluster_threshold=0.03)
    assert tight["families"] >= 2


@posix_only
def test_aggregate_by_condition(tmp_path):
    _write(tmp_path / "a.jsonl", "A", [WRONG])
    _write(tmp_path / "b.jsonl", "B", [CORRECT, CORRECT2])
    from macs.analysis.aggregate import analyse_dir
    sessions, summary = analyse_dir(tmp_path, backend="subprocess")
    assert summary["A"]["avg_correct"] == 0
    assert summary["B"]["avg_correct"] == 2
    assert summary["B"]["n_sessions"] == 1
