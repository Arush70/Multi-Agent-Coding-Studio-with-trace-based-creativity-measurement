"""Orchestrator tests with a FAKE model and the subprocess sandbox.

Arush - proves the whole pipeline wires together (learner -> tutor, and the full studio loop)
WITHOUT Ollama or Docker, so it runs in CI. The fake model returns known solutions, so the
produced traces must be schema-valid and contain the right events.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass

import pytest

from macs.tasks.bank import load_task, TASKS_ROOT
from macs.orchestrator import SessionConfig, run_session
from macs.trace.writer import validate_file, read_trace

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="subprocess sandbox rlimits differ on Windows; CI is Linux")

TASK = TASKS_ROOT / "tier1" / "t1_mbpp19_test_duplicate"

# three genuinely different correct implementations -> should form >1 family
IMPLS = [
    "def test_duplicate(a):\n    return len(a) != len(set(a))\n",
    "def test_duplicate(a):\n    seen = set()\n    for x in a:\n        if x in seen:\n            return True\n        seen.add(x)\n    return False\n",
    "def test_duplicate(a):\n    import collections\n    return any(v > 1 for v in collections.Counter(a).values())\n",
]


@dataclass
class FakeCompletion:
    text: str
    prompt_tokens: int = 3
    completion_tokens: int = 7
    request_hash: str = "fake"
    cached: bool = False
    latency_s: float = 0.0


class FakeProfile:
    @property
    def identity(self):
        return {"profile": "fake", "model": "fake", "revision": "0", "quantisation": "none", "temperature": 0.7}


class VariedClient:
    """Returns a different implementation each engineer call -> diverse proposals."""
    profile = FakeProfile()
    def __init__(self): self._i = 0
    def chat(self, messages, *, seed=None, **kw):
        s = messages[0]["content"].lower()
        if "engineer" in s:
            code = IMPLS[self._i % len(IMPLS)]; self._i += 1
            return FakeCompletion(text=f"```python\n{code}```")
        if "innovator" in s:
            return FakeCompletion(text=f"Strategy variant {self._i}: use a distinct data structure.")
        if "critic" in s:
            return FakeCompletion(text="Consider the empty-list edge case.")
        if "tutor" in s:
            return FakeCompletion(text=f"```python\n{IMPLS[0]}```")
        return FakeCompletion(text="I would keep a set of values I have seen.")


class IdenticalClient(VariedClient):
    """Always returns the SAME implementation -> proposals converge -> Facilitator should fire."""
    def chat(self, messages, *, seed=None, **kw):
        s = messages[0]["content"].lower()
        if "engineer" in s:
            return FakeCompletion(text=f"```python\n{IMPLS[0]}```")
        if "innovator" in s:
            return FakeCompletion(text="The same idea again.")
        if "critic" in s:
            return FakeCompletion(text="Looks fine.")
        return FakeCompletion(text="A set of seen values.")


def _events(path):
    return list(read_trace(path))


@posix_only
def test_condition_A_valid_trace(tmp_path):
    path = run_session(load_task(TASK), SessionConfig("A", seed=1, sandbox_backend="subprocess"), VariedClient(), out_dir=tmp_path)
    assert validate_file(path) == []
    roles = [e["role"] for e in _events(path)]
    assert "learner_1" in roles and "learner_2" not in roles


@posix_only
def test_condition_B_studio_produces_proposals(tmp_path):
    cfg = SessionConfig("B", seed=1, sandbox_backend="subprocess", k_proposals=5)
    path = run_session(load_task(TASK), cfg, VariedClient(), out_dir=tmp_path)
    assert validate_file(path) == []
    evs = _events(path)
    proposals = [e for e in evs if e["event"] == "proposal"]
    verifs = [e for e in evs if e["event"] == "verification"]
    assert len(proposals) == 5                      # five Innovator proposals
    assert len(verifs) >= 5                          # each implementation verified (+ the revision)
    assert any(e["event"] == "selection" for e in evs)   # families were clustered


@posix_only
def test_facilitator_fires_on_convergence(tmp_path):
    # identical proposals -> distances ~0 -> below tau -> at least one intervention
    cfg = SessionConfig("B", seed=1, sandbox_backend="subprocess", k_proposals=5, tau=0.5)
    path = run_session(load_task(TASK), cfg, IdenticalClient(), out_dir=tmp_path)
    assert validate_file(path) == []
    interventions = [e for e in _events(path) if e["event"] == "intervention"]
    assert len(interventions) >= 1
    assert interventions[0]["intervention_type"] in ("constraint", "dissent", "exclusion", "hybridisation")


@posix_only
def test_D_minus_disables_facilitator(tmp_path):
    # same converging inputs, but D_minus -> Facilitator off -> NO interventions
    cfg = SessionConfig("D_minus", seed=1, sandbox_backend="subprocess", k_proposals=5, tau=0.5)
    path = run_session(load_task(TASK), cfg, IdenticalClient(), out_dir=tmp_path)
    assert validate_file(path) == []
    assert not any(e["event"] == "intervention" for e in _events(path))
    roles = [e["role"] for e in _events(path)]
    assert "learner_2" in roles                     # D_minus is a two-learner condition
