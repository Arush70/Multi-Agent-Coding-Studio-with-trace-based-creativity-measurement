"""Tests for the task loader.

Arush - the important test here is test_heldout_never_in_agent_view. It is the
automated guarantee of your single most important rule. If someone (including future
you) refactors the loader and accidentally exposes held-out tests, this test goes red
and the mistake never reaches an experiment.
"""
from __future__ import annotations

from pathlib import Path
from macs.tasks.bank import load_task, load_bank, TASKS_ROOT

TIER1 = TASKS_ROOT / "tier1" / "t1_two_sum"


def test_loads_and_has_expected_fields():
    t = load_task(TIER1)
    assert t.id == "t1_two_sum"
    assert t.entry_point == "two_sum"
    assert len(t.known_families) >= 3          # a task with <3 strategies does not belong in the bank
    assert len(t.public_tests) >= 1


def test_heldout_never_in_agent_view():
    # The guarantee: no held-out test string appears in what an agent is shown.
    t = load_task(TIER1)
    view = t.agent_view
    for ht in t.heldout_tests():
        # compare on the distinctive part of each assert (the expected values)
        assert ht not in view, f"HELD-OUT TEST LEAKED INTO AGENT VIEW: {ht}"
    # and the object itself must not carry them as an attribute
    assert not hasattr(t, "heldout_tests_list")
    assert "heldout" not in t.__dict__


def test_scorer_can_still_reach_heldout():
    # The scorer (and only the scorer) can get them, on demand, from disk.
    t = load_task(TIER1)
    hts = t.heldout_tests()
    assert len(hts) >= 2
    assert any("range(10000)" in h for h in hts)   # the large-input case is present


def test_bank_loads_and_filters_by_tier():
    all_tasks = load_bank()
    assert len(all_tasks) >= 2
    tier1 = load_bank(tier=1)
    assert all(t.tier == 1 for t in tier1)
    tier2 = load_bank(tier=2)
    assert all(t.tier == 2 for t in tier2)
