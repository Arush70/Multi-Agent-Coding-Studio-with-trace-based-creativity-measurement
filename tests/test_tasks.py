"""Tests for the task loader and the provenance of the task bank.

Arush - the key guarantees here: (1) held-out tests never leak into what an agent sees,
for EVERY task; (2) every task carries a real source citation (your professor's requirement);
(3) every task admits at least three strategy families so novelty has something to detect.
"""
from __future__ import annotations
from macs.tasks.bank import load_task, load_bank, TASKS_ROOT

SAMPLE = TASKS_ROOT / "tier1" / "t1_mbpp19_test_duplicate"


def test_loads_and_has_expected_fields():
    t = load_task(SAMPLE)
    assert t.id == "t1_mbpp19_test_duplicate"
    assert t.entry_point == "test_duplicate"
    assert len(t.known_families) >= 3
    assert len(t.public_tests) >= 1


def test_every_task_has_a_citation():
    # Provenance: the professor asked that every problem trace to a paper/benchmark.
    for t in load_bank():
        assert t.citation, f"{t.id} has no citation"


def test_every_task_has_three_strategies():
    for t in load_bank():
        assert len(t.known_families) >= 3, f"{t.id} has < 3 families"


def test_heldout_never_in_agent_view_for_any_task():
    for t in load_bank():
        view = t.agent_view
        for ht in t.heldout_tests():
            assert ht not in view, f"HELD-OUT TEST LEAKED in {t.id}: {ht}"


def test_bank_loads_all_three_tiers():
    assert len(load_bank(tier=1)) >= 8
    assert len(load_bank(tier=2)) >= 4
    assert len(load_bank(tier=3)) >= 4          # the hard/open problems
    assert all(t.tier == 3 for t in load_bank(tier=3))


def test_tier3_is_heuristic_scored():
    for t in load_bank(tier=3):
        assert t.scoring == "heuristic"          # no single right answer; metric-scored
