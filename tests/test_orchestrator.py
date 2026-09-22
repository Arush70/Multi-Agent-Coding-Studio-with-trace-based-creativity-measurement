"""Orchestrator test with a FAKE model and the subprocess sandbox.

Arush - this proves the whole pipeline wires together (learner -> tutor -> sandbox ->
trace) WITHOUT needing Ollama or Docker, so it runs in CI. The fake model returns a known
correct two_sum, so the produced trace must be schema-valid and show 2/2 public tests
passing. If any wiring breaks, this goes red.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field

import pytest

from macs.tasks.bank import load_task, TASKS_ROOT
from macs.orchestrator import SessionConfig, run_session
from macs.trace.writer import validate_file, read_trace

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="subprocess sandbox rlimits are POSIX-only; on Windows this path still runs but CI is Linux")

CORRECT = "```python\ndef two_sum(nums, target):\n seen={}\n for i,v in enumerate(nums):\n  if target-v in seen: return [seen[target-v],i]\n  seen[v]=i\n```"


@dataclass
class FakeCompletion:
    text: str
    prompt_tokens: int = 3
    completion_tokens: int = 7
    request_hash: str = "fakehash"
    cached: bool = False
    latency_s: float = 0.0


class FakeProfile:
    @property
    def identity(self):
        return {"profile": "fake", "model": "fake-model", "revision": "0000",
                "quantisation": "none", "temperature": 0.7}


class FakeClient:
    """Returns tutor code when it sees the tutor system prompt, else a short learner line."""
    profile = FakeProfile()

    def chat(self, messages, *, seed=None, **kw):
        system = messages[0]["content"].lower()
        text = CORRECT if "tutor" in system else "I would keep a dict of complements I have seen."
        return FakeCompletion(text=text)


def _run(condition, tmp_path):
    task = load_task(TASKS_ROOT / "tier1" / "t1_two_sum")
    cfg = SessionConfig(condition=condition, seed=1, sandbox_backend="subprocess")
    return run_session(task, cfg, FakeClient(), out_dir=tmp_path)


@posix_only
def test_condition_A_produces_valid_trace(tmp_path):
    path = _run("A", tmp_path)
    assert validate_file(path) == []
    events = list(read_trace(path))
    roles = [e["role"] for e in events]
    assert "learner_1" in roles and "learner_2" not in roles      # A has exactly one learner
    ver = [e for e in events if e["event"] == "verification"][0]
    assert ver["tests_passed"] == 2 and ver["tests_failed"] == 0   # the correct code passes both public tests


@posix_only
def test_condition_C_has_two_learners(tmp_path):
    path = _run("C", tmp_path)
    assert validate_file(path) == []
    roles = [e["role"] for e in read_trace(path)]
    assert "learner_1" in roles and "learner_2" in roles           # C has two learners


def test_studio_conditions_not_yet_implemented(tmp_path):
    # Until weeks 3-4, asking for a studio condition must fail loudly, not silently do nothing.
    task = load_task(TASKS_ROOT / "tier1" / "t1_two_sum")
    with pytest.raises(NotImplementedError):
        run_session(task, SessionConfig(condition="D", seed=1, sandbox_backend="subprocess"), FakeClient(), out_dir=tmp_path)
