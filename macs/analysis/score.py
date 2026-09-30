"""Score finished traces on the HELD-OUT tests.

Arush - this is the step that decides 'is it correct?', and it runs AFTER the sessions, over
the saved traces - never during a session. That separation is deliberate: held-out tests live
only here and in the scorer, so they can never have reached a prompt. Only the scorer calls
task.heldout_tests().
"""
from __future__ import annotations

from pathlib import Path

from macs.tasks.bank import Task, load_bank
from macs.sandbox import runner
from macs.trace.writer import read_trace

_BANK: dict[str, Task] | None = None


def task_by_id(task_id: str) -> Task:
    global _BANK
    if _BANK is None:
        _BANK = {t.id: t for t in load_bank()}
    return _BANK[task_id]


def implementations(events: list[dict]) -> list[tuple[str, str]]:
    """Every code artefact in a trace, as (proposal_id, code). Tutor 'implementation',
    studio 'implementation' and 'revision' events all count."""
    out = []
    for e in events:
        if e.get("event") in ("implementation", "revision") and e.get("content"):
            out.append((e.get("proposal_id", ""), e["content"]))
    return out


def final_code(events: list[dict]) -> str | None:
    impls = implementations(events)
    return impls[-1][1] if impls else None


def score_code_heldout(task: Task, code: str, *, backend: str = "subprocess") -> dict:
    """Run one solution against the task's held-out tests. Returns pass/fail counts and whether
    it passed ALL of them (the definition of 'correct' for a test-scored task)."""
    tests = task.heldout_tests()
    if not tests:                       # heuristic (Tier 3) tasks have no held-out tests
        return {"scored": False, "passed_all": None, "n_pass": 0, "n_fail": 0}
    res = runner.run(code, tests, backend=backend, cpu_seconds=8)
    passed_all = res.error is None and res.tests_failed == 0 and res.tests_passed == len(tests)
    return {"scored": True, "passed_all": passed_all,
            "n_pass": res.tests_passed, "n_fail": res.tests_failed, "error": res.error}


def score_trace(path: str | Path, *, backend: str = "subprocess") -> dict:
    """Score every implementation in a trace on held-out tests. The session is 'correct' if its
    FINAL solution passes all held-out tests."""
    events = list(read_trace(path))
    start = next(e for e in events if e["event"] == "session_start")
    task = task_by_id(start["task_id"])
    impls = implementations(events)
    per = [score_code_heldout(task, code, backend=backend) for _, code in impls]
    final = per[-1] if per else {"scored": False, "passed_all": None}
    n_correct = sum(1 for p in per if p.get("passed_all"))
    return {
        "session_id": start["session_id"], "task_id": task.id, "condition": start["condition"],
        "n_proposals": len(impls), "n_correct_proposals": n_correct,
        "final_passed_heldout": final.get("passed_all"),
        "scoring": task.scoring,
        "per_proposal": per,
    }
