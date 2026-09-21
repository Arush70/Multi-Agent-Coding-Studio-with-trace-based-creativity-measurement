"""Tests for the trace schema and writer.

Arush - these guard the thing your entire results section is built on: the trace.
If the schema or writer silently changes shape, these fail, and you find out before
a single experiment runs on the bad format.
"""
from __future__ import annotations

import pytest
import jsonschema

from macs.trace.writer import TraceWriter, validate_file, validate_event

HEADER = {
    "model": "qwen2.5-coder:7b-instruct-q4_K_M",
    "revision": "dae161e27b0e",
    "quantisation": "q4_K_M",
    "temperature": 0.7,
    "seed": 1,
    "prompt_hashes": {"tutor": "abc123"},
}


def _write_valid_session(path):
    with TraceWriter(path, session_id="S1", task_id="t1_two_sum", condition="C", header=HEADER) as tr:
        tr.log("learner_1", "turn", content="loop over every pair")
        # action_tag is the field added after reading ASTRA - prove a tutor turn can carry it.
        tr.log("tutor", "turn", content="what does that cost?", action_tag="QUESTION")
        tr.log("tutor", "proposal", content="def two_sum(...): ...", proposal_id="p1")
        tr.log("verifier", "verification", content="ran tests", code_hash="deadbeef",
               tests_passed=2, tests_failed=0, test_set="public", runtime_s=0.01)


def test_valid_session_produces_valid_file(tmp_path):
    p = tmp_path / "S1.jsonl"
    _write_valid_session(p)
    assert validate_file(p) == []          # a clean session has zero errors


def test_astra_action_tag_is_accepted(tmp_path):
    # Superset check: every ASTRA action tag must validate on a turn event.
    for tag in ["QUESTION", "HINT", "EXPLANATION", "ENCOURAGEMENT",
                "INVITE_QUIET_MEMBER", "SUMMARISE", "MEDIATE_CONFLICT",
                "ENCOURAGE_COLLAB", "NONE"]:
        validate_event({
            "session_id": "S1", "task_id": "t", "condition": "D", "seq": 0,
            "ts": "2026-09-21T05:00:00+00:00", "role": "facilitator",
            "event": "turn", "action_tag": tag,
        })


def test_proposal_without_id_is_rejected():
    # A proposal MUST carry a proposal_id (schema conditional). This is the kind of
    # mistake that would quietly break family clustering if it slipped through.
    with pytest.raises(jsonschema.ValidationError):
        validate_event({
            "session_id": "S1", "task_id": "t", "condition": "D", "seq": 1,
            "ts": "2026-09-21T05:00:00+00:00", "role": "innovator",
            "event": "proposal", "content": "idea",   # missing proposal_id
        })


def test_unknown_field_is_rejected():
    # additionalProperties:false - a typo'd field name is caught, not silently stored.
    with pytest.raises(jsonschema.ValidationError):
        validate_event({
            "session_id": "S1", "task_id": "t", "condition": "D", "seq": 2,
            "ts": "2026-09-21T05:00:00+00:00", "role": "tutor", "event": "turn",
            "typo_field": "oops",
        })


def test_out_of_order_seq_is_flagged(tmp_path):
    # Hand-write a file with a seq that goes backwards and confirm validate_file catches it.
    p = tmp_path / "bad.jsonl"
    import json
    rows = [
        {"session_id": "S1", "task_id": "t", "condition": "A", "seq": 0,
         "ts": "2026-09-21T05:00:00+00:00", "role": "system", "event": "turn"},
        {"session_id": "S1", "task_id": "t", "condition": "A", "seq": 0,   # duplicate seq
         "ts": "2026-09-21T05:00:01+00:00", "role": "system", "event": "turn"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    errs = validate_file(p)
    assert any("monotonic" in e for e in errs)
