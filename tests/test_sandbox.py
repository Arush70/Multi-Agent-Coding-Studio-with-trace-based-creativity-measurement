"""Sandbox tests - now run on every OS (subprocess backend is cross-platform).

Arush - the loop/timeout guard is the important one: untrusted code that never returns
must be killed and reported as an error, not hang your experiment. On POSIX a CPU rlimit
also applies; on Windows the wall-clock timeout does the job.
"""
from __future__ import annotations
from macs.sandbox.runner import run

GOOD = "def two_sum(nums, t):\n seen={}\n for i,v in enumerate(nums):\n  if t-v in seen: return [seen[t-v],i]\n  seen[v]=i"
BAD  = "def two_sum(nums, t):\n return [0, 0]"


def test_correct_code_passes():
    r = run(GOOD, ["assert two_sum([2,7,11,15],9)==[0,1]", "assert two_sum([3,2,4],6)==[1,2]"], backend="subprocess")
    assert r.tests_passed == 2 and r.tests_failed == 0 and r.error is None


def test_wrong_answer_is_a_fail_not_an_error():
    r = run(BAD, ["assert two_sum([2,7,11,15],9)==[0,1]"], backend="subprocess")
    assert r.tests_passed == 0 and r.tests_failed == 1 and r.error is None


def test_syntax_error_loads_as_failure():
    r = run("def two_sum(:\n pass", ["assert two_sum([],0)==[]"], backend="subprocess")
    assert r.tests_passed == 0


def test_infinite_loop_is_caught_as_error():
    r = run("def f():\n while True: pass\nf()", ["assert True"], backend="subprocess", cpu_seconds=2)
    assert r.error is not None and r.tests_passed == 0


def test_as_trace_fields_shape():
    r = run(GOOD, ["assert two_sum([2,7,11,15],9)==[0,1]"], backend="subprocess")
    fields = r.as_trace_fields("public")
    assert set(fields) == {"code_hash", "tests_passed", "tests_failed", "test_set", "runtime_s"}
