"""Learner contribution (RQ3): how much of what the solver produced can be traced to the learners.

The learners speak in words and the solver writes code, so their contribution is measured on the
shared vocabulary of solution constructs that the Facilitator already uses (a set, a dictionary /
hash map, sorting, a heap, recursion, bit manipulation, nested loops; facilitator.CONSTRUCTS):

  * learner_constructs   constructs the learners mention in their turns (keyword match on text)
  * uptake_final         share of those constructs that the FINAL solution uses
                         (the last implementation or revision in the session)
  * uptake_any           share that ANY solution in the session uses
  * learner_led_share    share of the session's solutions that use at least one learner construct
  * final_from_learners  1 if the final solution uses a learner construct, else 0
  * learner_novel_share  share of the learners' constructs that the task's ordinary solutions
                         (the frozen reference set) rarely use (in under 20% of them): did the
                         learners bring an unusual idea?

All are None when the learners mention no construct. Pre-registered 2026-10-20 as exploratory
measures for RQ3; they describe the mechanism and are not part of the Holm family.
"""
from __future__ import annotations

import re

from macs.facilitator import code_constructs

# words a learner might use for each construct; matched case-insensitively on word boundaries
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "a set": ("set", "sets", "seen set", "hash set", "hashset"),
    "a dictionary / hash map": ("dict", "dictionary", "dictionaries", "hash map", "hashmap", "hash table",
                                "counter", "frequency", "frequencies", "tally", "count each", "counts of"),
    "sorting": ("sort", "sorted", "sorting", "order them", "in order"),
    "a heap": ("heap", "priority queue", "heapq"),
    "recursion": ("recursion", "recursive", "recursively", "recurse"),
    "bit manipulation": ("bit", "bits", "bitwise", "bitmask", "xor", "mask"),
    "nested loops": ("nested loop", "nested loops", "two loops", "double loop", "every pair", "each pair",
                     "all pairs", "compare every", "loop inside"),
}
_PATTERNS = {c: re.compile(r"\b(" + "|".join(re.escape(k) for k in ks) + r")\b", re.I) for c, ks in _KEYWORDS.items()}
RARE_IN_REFERENCE = 0.20


def text_constructs(text: str) -> set[str]:
    """Constructs mentioned in natural-language text."""
    return {c for c, p in _PATTERNS.items() if p.search(text or "")}


def learner_contribution(events: list[dict], reference_codes: list[str] | None = None) -> dict:
    """The measures above for one session's events (a trace read with read_trace)."""
    turns = [e.get("content", "") for e in events
             if str(e.get("role", "")).startswith("learner_") and e.get("event") == "turn"]
    codes = [e["content"] for e in events if e.get("event") in ("implementation", "revision") and e.get("content")]
    mentioned: set[str] = set()
    for t in turns:
        mentioned |= text_constructs(t)
    out = {"n_learner_turns": len(turns), "learner_constructs": sorted(mentioned), "uptake_final": None,
           "uptake_any": None, "learner_led_share": None, "final_from_learners": None,
           "learner_novel_share": None}
    if not mentioned or not codes:
        return out
    used = [code_constructs(c) for c in codes]
    final = used[-1]
    any_used = set().union(*used)
    out["uptake_final"] = len(mentioned & final) / len(mentioned)
    out["uptake_any"] = len(mentioned & any_used) / len(mentioned)
    out["learner_led_share"] = sum(1 for u in used if u & mentioned) / len(used)
    out["final_from_learners"] = int(bool(final & mentioned))
    if reference_codes:
        ref_used = [code_constructs(c) for c in reference_codes]
        rate = {c: sum(1 for u in ref_used if c in u) / len(ref_used) for c in mentioned}
        out["learner_novel_share"] = sum(1 for c in mentioned if rate[c] < RARE_IN_REFERENCE) / len(mentioned)
    return out
