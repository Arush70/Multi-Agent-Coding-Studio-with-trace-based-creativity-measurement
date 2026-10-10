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


# --- added 2026-10-07 (docs/DEV_LOG.md): checks for the fixes made that day -----------------

class RecordingClient(VariedClient):
    """Remembers every prompt it was sent, so a test can check what each role actually saw."""
    def __init__(self):
        super().__init__(); self.seen = []
    def chat(self, messages, *, seed=None, **kw):
        self.seen.append(messages)
        if "learner in an introductory" in messages[0]["content"]:
            return FakeCompletion(text=f"LEARNER-IDEA-{len(self.seen)}")
        return super().chat(messages, seed=seed, **kw)


def _user_prompts(client, role_word):
    return [m[1]["content"] for m in client.seen if role_word in m[0]["content"].lower().split("\n")[0]]


@posix_only
def test_learner_turns_reach_the_tutor(tmp_path):
    c = RecordingClient()
    run_session(load_task(TASK), SessionConfig("C", seed=1, sandbox_backend="subprocess"), c, out_dir=tmp_path)
    tutor_prompt = _user_prompts(c, "programming tutor")[0]
    assert tutor_prompt.count("LEARNER-IDEA-") == 2          # both learners' turns are shown
    second_learner = [m[1]["content"] for m in c.seen if "learner in an introductory" in m[0]["content"]][1]
    assert "LEARNER-IDEA-" in second_learner                  # learner 2 sees learner 1


@posix_only
def test_learner_turns_reach_the_innovator(tmp_path):
    c = RecordingClient()
    run_session(load_task(TASK), SessionConfig("B", seed=1, sandbox_backend="subprocess", tau=0.5), c, out_dir=tmp_path)
    assert all("LEARNER-IDEA-" in p for p in _user_prompts(c, "innovator"))


@posix_only
def test_critic_and_revision_see_the_code(tmp_path):
    c = RecordingClient()
    run_session(load_task(TASK), SessionConfig("B", seed=1, sandbox_backend="subprocess", tau=0.5), c, out_dir=tmp_path)
    assert "def test_duplicate" in _user_prompts(c, "critic")[0]
    revision_prompt = [p for p in _user_prompts(c, "engineer") if "Revise it" in p][0]
    assert "def test_duplicate" in revision_prompt


def test_degenerate_reference_tau_falls_back(monkeypatch):
    import numpy as np
    from macs import orchestrator
    from macs.metrics import reference

    class R:  # a reference made of 10 identical solutions -> every NN distance is 0 -> tau 0
        X_ref = np.ones((10, 26))
    monkeypatch.setattr(reference, "build_reference_stats", lambda *a, **k: R())
    tau, source = SessionConfig("D", seed=1).resolve_tau(load_task(TASK))
    assert tau == orchestrator.DEFAULT_TAU and source == "default:degenerate_reference"


@posix_only
def test_header_records_tau_and_learners(tmp_path):
    path = run_session(load_task(TASK), SessionConfig("D", seed=1, sandbox_backend="subprocess", tau=0.5),
                       VariedClient(), out_dir=tmp_path)
    h = next(e for e in _events(path) if e["event"] == "session_start")["header"]
    assert h["tau"] == 0.5 and h["tau_source"] == "override" and h["n_learners"] == 2
    assert h["max_revision_rounds"] == 3 and h["orchestrator_version"] == "0.2.0"


# --- added 2026-10-09 (docs/DEV_LOG.md): selection, revision rounds, token ceiling, Facilitator ---

class SelectingClient(VariedClient):
    """The selector names p2; everything else as VariedClient."""
    def __init__(self, reply="SELECTED: p2\nIt uses a different structure."):
        super().__init__(); self.reply = reply; self.seen = []
    def chat(self, messages, *, seed=None, **kw):
        self.seen.append(messages)
        if "acting as the selector" in messages[0]["content"]:
            return FakeCompletion(text=self.reply)
        return super().chat(messages, seed=seed, **kw)


@posix_only
def test_critic_selects_a_proposal_and_revision_builds_on_it(tmp_path):
    c = SelectingClient()
    path = run_session(load_task(TASK), SessionConfig("B", seed=1, sandbox_backend="subprocess", tau=0.5), c, out_dir=tmp_path)
    assert validate_file(path) == []
    evs = _events(path)
    sel = next(e for e in evs if e["event"] == "selection")
    assert sel["proposal_id"] == "p2" and "(critic)" in sel["content"]
    assert next(e for e in evs if e["event"] == "critique")["proposal_id"] == "p2"
    assert next(e for e in evs if e["event"] == "revision")["parent_ids"] == ["p2"]
    sel_prompt = next(m[1]["content"] for m in c.seen if "acting as the selector" in m[0]["content"])
    assert all(f"[p{k}]" in sel_prompt for k in range(5))          # the selector saw all five


@posix_only
def test_selection_falls_back_when_reply_is_unusable(tmp_path):
    c = SelectingClient(reply="I like the third one.")
    path = run_session(load_task(TASK), SessionConfig("B", seed=1, sandbox_backend="subprocess", tau=0.5), c, out_dir=tmp_path)
    sel = next(e for e in _events(path) if e["event"] == "selection")
    assert "fallback" in sel["content"] and sel["proposal_id"] == "p0"   # all pass -> earliest


class FailingRevisionClient(VariedClient):
    """Revisions are always wrong -> the loop must run the full 3 rounds, then stop."""
    def chat(self, messages, *, seed=None, **kw):
        if "Revise it" in messages[-1]["content"]:
            return FakeCompletion(text="```python\ndef test_duplicate(a):\n    return None\n```")
        return super().chat(messages, seed=seed, **kw)


@posix_only
def test_revision_rounds_stop_when_passing_and_cap_at_three(tmp_path):
    ok = run_session(load_task(TASK), SessionConfig("B", seed=1, sandbox_backend="subprocess", tau=0.5),
                     VariedClient(), out_dir=tmp_path)
    assert sum(e["event"] == "revision" for e in _events(ok)) == 1        # passed -> stop after 1
    bad = run_session(load_task(TASK), SessionConfig("B", seed=1, sandbox_backend="subprocess", tau=0.5),
                      FailingRevisionClient(), out_dir=tmp_path)
    revs = [e for e in _events(bad) if e["event"] == "revision"]
    assert [r["proposal_id"] for r in revs] == ["p_rev1", "p_rev2", "p_rev3"]
    assert [r["parent_ids"][0] for r in revs[1:]] == ["p_rev1", "p_rev2"]  # each builds on the last
    assert validate_file(bad) == []


@posix_only
def test_token_ceiling_stops_the_session(tmp_path):
    # each fake call costs 10 tokens; a ceiling of 35 allows 4 calls, then stops cleanly
    cfg = SessionConfig("B", seed=1, sandbox_backend="subprocess", tau=0.5)
    cfg.token_ceiling = 35
    path = run_session(load_task(TASK), cfg, VariedClient(), out_dir=tmp_path)
    assert validate_file(path) == []
    evs = _events(path)
    assert evs[-1]["event"] == "session_end" and evs[-1]["content"].startswith("stopped")
    assert sum(1 for e in evs if e.get("request_hash")) == 4
    full = run_session(load_task(TASK), SessionConfig("A", seed=1, sandbox_backend="subprocess"), VariedClient(), out_dir=tmp_path)
    assert _events(full)[-1]["content"].startswith("completed")


@posix_only
def test_facilitator_can_fire_after_three_proposals_with_specific_instruction(tmp_path):
    cfg = SessionConfig("B", seed=1, sandbox_backend="subprocess", k_proposals=5, tau=0.5)
    path = run_session(load_task(TASK), cfg, IdenticalClient(), out_dir=tmp_path)
    evs = _events(path)
    first = next(i for i, e in enumerate(evs) if e["event"] == "intervention")
    proposals_before = sum(1 for e in evs[:first] if e["event"] == "proposal")
    assert proposals_before == 3                                   # fires after p2, shapes p3 and p4
    assert "does NOT use a set" in evs[first]["content"]             # names what they share (IMPLS[0] uses set())


def test_facilitator_window_counts_proposals():
    import numpy as np
    from macs.facilitator import Facilitator
    f = Facilitator(tau=0.5, window=3)
    for v in ([1, 0], [1, 0], [1, 0]):
        f.observe(np.array(v, float))
    assert f.window_metric() == 0.0 and f.should_intervene()[0]   # three proposals are enough


def test_facilitator_names_shared_construct_and_approach():
    from macs.facilitator import dominant_construct, approach_name, Facilitator
    codes = ["def f(a):\n    return len(set(a)) < len(a)\n", "def f(a):\n    s = set()\n    return s\n",
             "def f(a):\n    return sorted(a)\n"]
    assert dominant_construct(codes) == "a set"
    assert dominant_construct(["x = 1"]) is None
    assert approach_name("**Hash set scan**: keep a set of seen values. O(n).") == "Hash set scan"
    f = Facilitator(tau=0.1)
    kind, args = f.intervene(construct="a set", approach="Hash set scan")
    assert kind == "constraint" and args == {"excluded": "a set"}
    f.next_intervention()                                          # skip dissent
    kind, args = f.intervene(construct="a set", approach="Hash set scan")
    assert kind == "exclusion" and args["family"] == "Hash set scan, built on a set"
