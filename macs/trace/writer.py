"""Append-only JSONL trace writer with per-line validation.

Every event (a proposal, an intervention, a test result) is written as one JSON line, and
each line is validated against schema.json before it is written, so a malformed event fails
immediately with a clear error instead of surfacing later during analysis.

Design choices:
  * Append-only. Lines are never rewritten; the file is an immutable, ordered record of
    what happened.
  * seq is monotonic and validated on read, because the analysis depends on event order
    (e.g. "first 10 proposals").
  * The header (written once, on session_start) carries the model, revision, seed and
    prompt hashes: everything needed to reproduce the session.
  * session_end can carry a note (end_note), e.g. tokens used or why a session stopped.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import jsonschema

SCHEMA_PATH = Path(__file__).with_name("schema.json")

# Bumped when the schema changes shape. It is stamped into every session header, so a
# trace always says which schema version produced it. Frozen as the git tag
# schema-v0.1.0; any later schema change is a deliberate, versioned amendment.
# 0.2.0 (2026-10-20): conditions E, E_minus, S and P added to the condition enum (additive; every
# 0.1.0 trace is still valid).
# 0.3.0 (2026-10-23): conditions C4 and D4 and roles learner_3 to learner_6 (up to six learners).
SCHEMA_VERSION = "0.3.0"

_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)


def validate_event(event: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if the event does not conform to the schema."""
    _VALIDATOR.validate(event)


class TraceWriter:
    """Use as a context manager. It writes session_start on enter and session_end on
    exit, so every session is bracketed:

        with TraceWriter(path, session_id="S1", task_id="t1_two_sum",
                         condition="C", header=hdr) as tr:
            tr.log("learner_1", "turn", content="I'd loop over pairs")
            tr.log("tutor", "turn", content="What is the cost of that?", action_tag="QUESTION")
    """

    def __init__(self, path: str | Path, *, session_id: str, task_id: str, condition: str, header: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.task_id = task_id
        self.condition = condition
        self.seq = 0
        self._fh = None
        self._header = dict(header)
        # If the caller forgot the schema version, stamp the current one so every
        # header is complete.
        self._header.setdefault("schema_version", SCHEMA_VERSION)
        # Optional text for the session_end event (e.g. why the session stopped, tokens used).
        # Set it any time before the 'with' block exits.
        self.end_note: str | None = None

    def __enter__(self) -> "TraceWriter":
        self._fh = self.path.open("a", encoding="utf-8")
        self.log("system", "session_start", header=self._header)
        return self

    def __exit__(self, *exc) -> None:
        self.log("system", "session_end", content=self.end_note)
        if self._fh:
            self._fh.close()

    def log(self, role: str, event: str, **fields: Any) -> dict[str, Any]:
        # The five identity fields are added automatically so a caller only ever
        # supplies what is specific to the event. seq increments here, so ordering is
        # never the caller's job to get right.
        rec: dict[str, Any] = {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "condition": self.condition,
            "seq": self.seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "event": event,
        }
        # None-valued fields are dropped rather than written as null - schema.json is
        # strict (additionalProperties:false), so only real values go on the line.
        rec.update({k: v for k, v in fields.items() if v is not None})
        validate_event(rec)                       # <-- fail loudly BEFORE writing
        assert self._fh is not None, "TraceWriter must be used as a context manager (use 'with')"
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()                          # flush per line: a crash mid-session still leaves a readable trace
        self.seq += 1
        return rec


def read_trace(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield each event dict from a trace file, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def validate_file(path: str | Path) -> list[str]:
    """Check a whole trace file. Returns a list of human-readable error strings, empty
    if the file is completely valid. Used by the run scripts and the tests, so a bad
    trace cannot quietly enter the dataset."""
    errors: list[str] = []
    last_seq = -1
    for i, ev in enumerate(read_trace(path)):
        try:
            validate_event(ev)
        except jsonschema.ValidationError as e:
            errors.append(f"line {i}: {e.message}")
        if ev.get("seq", -1) <= last_seq:
            errors.append(f"line {i}: seq not monotonic (order is corrupted)")
        last_seq = ev.get("seq", last_seq)
    return errors
