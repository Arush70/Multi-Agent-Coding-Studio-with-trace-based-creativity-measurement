"""Learner-contribution measures (RQ3)."""
from macs.metrics.learners import learner_contribution, text_constructs


def _ev(turns, codes):
    ev = [{"role": f"learner_{i}", "event": "turn", "content": t} for i, t in enumerate(turns, start=1)]
    ev += [{"role": "engineer", "event": "implementation", "content": c} for c in codes]
    return ev


SET = "def f(a):\n    return len(set(a)) != len(a)\n"
SORT = "def f(a):\n    b = sorted(a)\n    return any(x == y for x, y in zip(b, b[1:]))\n"
PAIRS = "def f(a):\n    for i in range(len(a)):\n        for j in range(i + 1, len(a)):\n            if a[i] == a[j]:\n                return True\n    return False\n"


def test_text_constructs_reads_plain_language():
    assert text_constructs("I'd keep a set of seen values") == {"a set"}
    assert text_constructs("Sort the list, then compare neighbours") == {"sorting"}
    assert text_constructs("compare every pair with two loops") == {"nested loops"}
    assert text_constructs("I am not sure where to start") == set()
    assert "a set" not in text_constructs("I will reset the counter variable")     # word boundary


def test_uptake_and_led_share():
    r = learner_contribution(_ev(["Use a set of seen values.", "Or sort it first?"], [PAIRS, SET, SORT]))
    assert r["learner_constructs"] == ["a set", "sorting"]
    assert r["uptake_final"] == 0.5 and r["final_from_learners"] == 1      # final uses sorting
    assert r["uptake_any"] == 1.0
    assert abs(r["learner_led_share"] - 2 / 3) < 1e-9


def test_learner_novel_share_against_reference():
    ref = [SET] * 9 + [SORT]          # sorting appears in 10% of ordinary solutions -> rare
    r = learner_contribution(_ev(["Use a set.", "Sort it."], [SORT]), reference_codes=ref)
    assert r["learner_novel_share"] == 0.5


def test_no_learner_constructs_gives_none():
    r = learner_contribution(_ev(["Hmm, not sure."], [SET]))
    assert r["uptake_final"] is None and r["learner_led_share"] is None
