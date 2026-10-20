import sys

import pytest

from macs.analysis.mutation import mutants, mutation_score
from macs.analysis.score import task_by_id

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="subprocess sandbox rlimits differ on Windows")

CODE = "def test_duplicate(a):\n    seen = set()\n    for x in a:\n        if x in seen:\n            return True\n        seen.add(x)\n    return False\n"


def test_mutants_change_one_thing_each():
    ms = mutants("def f(a, b):\n    if a < b and a > 0:\n        return a + 1\n    return not b\n")
    ops = {op for op, _ in ms}
    assert {"comparison", "arithmetic", "constant", "boolean", "return"} <= ops
    assert all(src != "def f(a, b):\n    if a < b and a > 0:\n        return a + 1\n    return not b" for _, src in ms)
    assert any("a <= b" in src for _, src in ms) and any(" or " in src for _, src in ms)


@posix_only
def test_held_out_tests_kill_most_mutants():
    r = mutation_score(task_by_id("t1_mbpp19_test_duplicate"), CODE)
    assert r["total"] >= 3 and r["score"] >= 0.5
