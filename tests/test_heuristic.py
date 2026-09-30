"""Tier 3 heuristic scoring: metric vs baseline, feasibility, and the analysis integration.

Arush - checks the part that gives your hard/open problems real numbers: the baseline scores
exactly 1.0 against itself (sanity), an infeasible heuristic scores 0 (not a crash), and a Tier 3
session flows through session_metrics with a heuristic score attached. No model is used.
"""
from __future__ import annotations
import sys
import pytest

from macs.analysis.score import task_by_id
from macs.analysis.heuristic_score import score_heuristic_code
from macs.trace.writer import TraceWriter
from macs.analysis.aggregate import session_metrics

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="subprocess sandbox rlimits differ on Windows; CI is Linux")

HEADER = {"model": "fake", "revision": "0", "quantisation": "none", "temperature": 0.7,
          "seed": 1, "prompt_hashes": {"x": "y"}}

BEST_FIT = "def priority(item, bins):\n    return [-(b - item) for b in bins]\n"      # the baseline itself
WORST_FIT = "def priority(item, bins):\n    return [b - item for b in bins]\n"         # a worse, valid heuristic
INFEASIBLE = "def priority(item, bins):\n    return 5\n"                                # wrong return shape


@posix_only
def test_baseline_scores_one():
    t = task_by_id("t3_bin_packing")
    r = score_heuristic_code(t, BEST_FIT, backend="subprocess")
    assert r["ok"] and r["valid"]
    assert abs(r["score"] - 1.0) < 1e-9        # baseline vs itself is exactly 1.0


@posix_only
def test_worse_heuristic_scores_below_one():
    t = task_by_id("t3_bin_packing")
    r = score_heuristic_code(t, WORST_FIT, backend="subprocess")
    assert r["ok"] and r["valid"]
    assert r["score"] < 1.0                     # worst-fit uses more bins than best-fit


@posix_only
def test_infeasible_scores_zero_not_crash():
    t = task_by_id("t3_bin_packing")
    r = score_heuristic_code(t, INFEASIBLE, backend="subprocess")
    assert r["ok"] is True                      # it ran...
    assert r["valid"] is False                  # ...but produced an infeasible answer
    assert r["score"] == 0.0
    assert r["error"] is None                   # infeasible is DATA, not a rig failure


@posix_only
def test_cap_set_rejects_a_line():
    t = task_by_id("t3_cap_set")
    # three vectors on a line (they sum to zero mod 3) -> not a valid cap set
    line = "def build_cap_set(n):\n    z=tuple([0]*n); a=tuple([1]+[0]*(n-1)); b=tuple([2]+[0]*(n-1))\n    return [z,a,b]\n"
    r = score_heuristic_code(t, line, backend="subprocess")
    assert r["valid"] is False and r["score"] == 0.0


@posix_only
def test_tier3_session_metrics(tmp_path):
    p = tmp_path / "s.jsonl"
    with TraceWriter(p, session_id="s", task_id="t3_bin_packing", condition="B", header=HEADER) as tr:
        tr.log("engineer", "implementation", content=BEST_FIT, proposal_id="p0")
        tr.log("engineer", "implementation", content=WORST_FIT, proposal_id="p1")
    m = session_metrics(p, backend="subprocess")
    assert m["scoring"] == "heuristic"
    assert m["n_correct"] == 2                  # both are feasible
    assert m["heuristic_score"] is not None     # a mean score vs baseline was computed
    assert m["families"] >= 1
