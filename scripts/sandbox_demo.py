"""Run a correct and a wrong solution through the sandbox against a real task's PUBLIC tests.

Arush - this ties the sandbox to the task loader:

    python scripts/sandbox_demo.py                 # subprocess backend (works now)
    python scripts/sandbox_demo.py --docker        # docker backend (after you install Docker)

You should see the correct solution pass all public tests and the wrong one fail - and
in both cases error=None, because a wrong answer is a result, not a crash.
"""
import sys
from macs.tasks.bank import load_task
from macs.sandbox.runner import run

BACKEND = "docker" if "--docker" in sys.argv else "subprocess"

CORRECT = "def test_duplicate(a):\n    return len(a) != len(set(a))\n"
WRONG = "def test_duplicate(a):\n    return True\n"

def main() -> None:
    task = load_task("tasks/tier1/t1_mbpp19_test_duplicate")
    tests = task.public_tests
    print(f"backend={BACKEND}  task={task.id}  public_tests={len(tests)}\n")
    for label, src in [("CORRECT", CORRECT), ("WRONG", WRONG)]:
        r = run(src, tests, backend=BACKEND, cpu_seconds=5)
        print(f"{label:<8} passed={r.tests_passed}/{r.tests_passed+r.tests_failed}  "
              f"error={r.error}  runtime={r.runtime_s:.3f}s  hash={r.code_hash}")

if __name__ == "__main__":
    main()
