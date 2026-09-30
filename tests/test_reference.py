"""Reference-set loading/fitting tests - no model needed.

Arush - proves the loader turns saved solutions into a working novelty yardstick, so the
whole novelty path is tested without running the model (which CI can't do).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from macs.metrics.reference import load_reference_solutions, build_reference_stats

# a handful of ordinary, varied solutions to a toy task
SOLUTIONS = [
    "def f(n):\n    return sum(range(n+1))",
    "def f(n):\n    s = 0\n    for i in range(n+1):\n        s += i\n    return s",
    "def f(n):\n    return n*(n+1)//2",
    "def f(n):\n    total = 0\n    i = 1\n    while i <= n:\n        total += i\n        i += 1\n    return total",
    "def f(n):\n    return sum(i for i in range(1, n+1))",
    "def f(n):\n    acc = 0\n    for k in range(1, n+1):\n        acc = acc + k\n    return acc",
]


def _write(tmp_path, task_id="t_demo"):
    p = tmp_path / f"{task_id}.jsonl"
    p.write_text("\n".join(json.dumps({"task_id": task_id, "seed": i, "code": c})
                           for i, c in enumerate(SOLUTIONS)), encoding="utf-8")
    return tmp_path


def test_load_reads_all_solutions(tmp_path):
    root = _write(tmp_path)
    assert len(load_reference_solutions("t_demo", root=root)) == len(SOLUTIONS)


def test_build_stats_gives_working_yardstick(tmp_path):
    root = _write(tmp_path)
    rs = build_reference_stats("t_demo", representation="ast", k=3, root=root)
    assert rs.X_ref.shape[0] == len(SOLUTIONS)          # one row per solution
    # a solution from the same family scores near-typical; a wildly different one scores higher
    from macs.metrics.represent import ast_feature_matrix, normalise
    typical = ast_feature_matrix([normalise("def f(n):\n    return sum(range(n+1))")])
    weird = ast_feature_matrix([normalise(
        "def f(n):\n    import math\n    return int(math.pow(n,2)/2 + n/2) if n%2==0 else (n*n+n)//2")])
    assert rs.novelty(weird)[0] >= rs.novelty(typical)[0]


def test_missing_reference_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reference_solutions("does_not_exist", root=tmp_path)
