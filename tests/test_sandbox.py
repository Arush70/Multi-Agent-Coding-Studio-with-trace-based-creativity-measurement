"""Sandbox tests.

Arush - these run in SUBPROCESS mode, because GitHub's CI is not guaranteed to have
Docker and I want these green everywhere. The Docker backend you verify by hand once
Docker is installed (scripts/sandbox_demo.py --docker). The behaviours tested here -
correct passes, wrong fails, a crash is an error not a silent fail, a timeout is caught
- are identical across both backends; only the isolation strength differs.
"""
from __future__ import annotations
import sys
import pytest
from macs.sandbox.runner import run

# resource limits used by subprocess mode are POSIX-only; skip on Windows CI if ever used there
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="subprocess resource limits are POSIX-only")

GOOD = "def two_sum(nums, t):\n seen={}\n for i,v in enumerate(nums):\n  if t-v in seen: return [seen[t-v],i]\n  seen[v]=i"
BAD  = "def two_sum(nums, t):\n return [0, 0]"   # wrong answer, but runs fine

@posix_only
def test_correct_code_passes():
    r = run(GOOD, ["assert two_sum([2,7,11,15],9)==[0,1]", "assert two_sum([3,2,4],6)==[1,2]"], backend="subprocess")
    assert r.tests_passed == 2 and r.tests_failed == 0
    assert r.error is None                       # passing code has no error

@posix_only
def test_wrong_answer_is_a_fail_not_an_error():
    r = run(BAD, ["assert two_sum([2,7,11,15],9)==[0,1]"], backend="subprocess")
    assert r.tests_passed == 0 and r.tests_failed == 1
    assert r.error is None                       # a wrong answer is DATA, not a rig error

@posix_only
def test_syntax_error_loads_as_failure():
    r = run("def two_sum(:\n pass", ["assert two_sum([],0)==[]"], backend="subprocess")
    assert r.tests_passed == 0                    # cannot even load -> every test fails

@posix_only
def test_infinite_loop_is_caught_as_error():
    # The single most important guard: untrusted code that never returns must be killed
    # and reported as an error, not hang your experiment forever.
    r = run("def f():\n while True: pass\nf()", ["assert True"], backend="subprocess", cpu_seconds=2)
    assert r.error is not None                    # killed by CPU limit / timeout
    assert r.tests_passed == 0

@posix_only
def test_as_trace_fields_shape():
    r = run(GOOD, ["assert two_sum([2,7,11,15],9)==[0,1]"], backend="subprocess")
    fields = r.as_trace_fields("public")
    assert set(fields) == {"code_hash", "tests_passed", "tests_failed", "test_set", "runtime_s"}
    assert fields["test_set"] == "public"
