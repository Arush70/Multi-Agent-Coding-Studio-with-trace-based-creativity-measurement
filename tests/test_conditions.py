"""The extended conditions (E, E_minus, S, P) and the condition registry."""
from __future__ import annotations

import pytest

from macs import conditions as conds
from macs.orchestrator import (SessionConfig, run_session, split_approaches, is_comparable,
                               SAMPLING_TOKEN_TARGET)
from macs.trace.writer import validate_file, read_trace
from tests.test_orchestrator import TASK, IMPLS, FakeCompletion, VariedClient, IdenticalClient, posix_only
from macs.tasks.bank import load_task


def _ev(path):
    return list(read_trace(path))


def test_registry_covers_every_condition_once():
    assert conds.CORE == ["A", "B", "C", "D", "D_minus"]
    assert set(conds.ALL) == {"A", "B", "C", "D", "D_minus", "E", "E_minus", "S", "P"}
    assert conds.get("E").facilitator and not conds.get("E").scaffold
    assert not conds.get("E_minus").facilitator and not conds.get("E_minus").scaffold
    assert conds.get("D").scaffold and conds.get("D_minus").scaffold
    assert {c: conds.get(c).n_learners for c in conds.ALL} == {
        "A": 1, "B": 1, "C": 2, "D": 2, "D_minus": 2, "E": 2, "E_minus": 2, "S": 2, "P": 2}
    with pytest.raises(ValueError):
        conds.get("Z")


def test_version_gate_keeps_unchanged_pilot_traces():
    assert is_comparable("0.2.3") and is_comparable("0.3.0")
    assert not is_comparable("0.2.2") and not is_comparable(None)


class RecordingClient(VariedClient):
    """Records the Innovator's system prompt and user message."""
    def __init__(self):
        super().__init__()
        self.innovator_calls = []

    def chat(self, messages, *, seed=None, **kw):
        if "innovator" in messages[0]["content"].lower():
            self.innovator_calls.append((messages[0]["content"], messages[1]["content"]))
        return super().chat(messages, seed=seed, **kw)


@posix_only
def test_neutral_innovator_sees_no_history_and_no_instruction_to_differ(tmp_path):
    task = load_task(TASK)
    c = RecordingClient()
    p = run_session(task, SessionConfig(condition="E_minus", seed=1, sandbox_backend="subprocess", tau=0.0), c,
                    out_dir=tmp_path)
    assert validate_file(p) == []
    assert len(c.innovator_calls) == 5
    for system, user in c.innovator_calls:
        assert "differs" not in system and "already" not in system
        assert "Strategies proposed so far" not in user
    hdr = _ev(p)[0]["header"]
    assert hdr["scaffold"] is False and hdr["facilitator_enabled"] is False and hdr["n_learners"] == 2


@posix_only
def test_scaffolded_innovator_still_sees_history(tmp_path):
    c = RecordingClient()
    run_session(load_task(TASK), SessionConfig(condition="D", seed=1, sandbox_backend="subprocess", tau=0.0), c,
                out_dir=tmp_path)
    assert all("Strategies proposed so far" in u for _, u in c.innovator_calls)


@posix_only
def test_E_facilitator_fires_on_converging_proposals_and_E_minus_does_not(tmp_path):
    task = load_task(TASK)
    on = run_session(task, SessionConfig(condition="E", seed=1, sandbox_backend="subprocess", tau=0.5),
                     IdenticalClient(), out_dir=tmp_path)
    off = run_session(task, SessionConfig(condition="E_minus", seed=1, sandbox_backend="subprocess", tau=0.5),
                      IdenticalClient(), out_dir=tmp_path)
    assert sum(e["event"] == "intervention" for e in _ev(on)) >= 1
    assert sum(e["event"] == "intervention" for e in _ev(off)) == 0


class CountingTutor(VariedClient):
    """Tutor replies cost 1,000 tokens each; records temperatures and seeds."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def chat(self, messages, *, seed=None, **kw):
        if "tutor" in messages[0]["content"].lower():
            self.calls.append((seed, kw.get("temperature")))
            return FakeCompletion(text=f"```python\n{IMPLS[len(self.calls) % 3]}```", prompt_tokens=400,
                                  completion_tokens=600)
        return super().chat(messages, seed=seed, **kw)


@posix_only
def test_sampling_baseline_samples_to_the_token_budget(tmp_path):
    c = CountingTutor()
    p = run_session(load_task(TASK), SessionConfig(condition="S", seed=2, sandbox_backend="subprocess"), c,
                    out_dir=tmp_path)
    ev = _ev(p)
    assert validate_file(p) == []
    impls = [e for e in ev if e["event"] == "implementation"]
    assert len(impls) == len(c.calls) >= 10
    assert len({s for s, _ in c.calls}) == len(c.calls)          # a new seed every time
    assert all(temp == 1.0 for _, temp in c.calls)
    used = sum(e.get("prompt_tokens", 0) + e.get("completion_tokens", 0) for e in ev)
    assert SAMPLING_TOKEN_TARGET <= used < SAMPLING_TOKEN_TARGET + 1_000 + 200
    assert ev[0]["header"]["solver"] == "sampling"


def test_split_approaches_pairs_each_block_with_its_prose():
    text = ("Approach 1: set length, O(n).\n```python\ndef f(a):\n    return 1\n```\n"
            "Approach 2: sorting, O(n log n).\n```py\ndef f(a):\n    return 2\n```\n"
            "Approach 3: counter.\n```python\ndef f(a):\n    return 3\n")       # last block unclosed
    parts = split_approaches(text)
    assert [c.strip()[-1] for _, c in parts] == ["1", "2", "3"]
    assert "set length" in parts[0][0] and "sorting" in parts[1][0]


class SinglePromptClient(VariedClient):
    def chat(self, messages, *, seed=None, **kw):
        if "approaches" in messages[0]["content"].lower():
            body = "".join(f"Approach {i + 1}: variant {i}.\n```python\n{IMPLS[i % 3]}```\n" for i in range(5))
            return FakeCompletion(text=body, prompt_tokens=200, completion_tokens=900)
        return super().chat(messages, seed=seed, **kw)


@posix_only
def test_single_prompt_baseline_logs_k_proposals_from_one_call(tmp_path):
    p = run_session(load_task(TASK), SessionConfig(condition="P", seed=1, sandbox_backend="subprocess"),
                    SinglePromptClient(), out_dir=tmp_path)
    ev = _ev(p)
    assert validate_file(p) == []
    assert sum(e["event"] == "proposal" for e in ev) == 5
    assert sum(e["event"] == "implementation" for e in ev) == 5
    assert sum(e["event"] == "verification" for e in ev) == 5
    # the one call's tokens are counted once
    assert sum(e.get("completion_tokens", 0) for e in ev if e["role"] == "tutor") == 900
