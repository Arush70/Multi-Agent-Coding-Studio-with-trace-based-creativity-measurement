"""Recorded code-execution verdicts, so rescoring the stored traces gives the submitted numbers.

Correctness (Tier 1/2: passed all held-out tests) and the Tier 3 heuristic score are found by
running each proposal's code under a CPU-time limit. A program that finishes close to that limit
can pass on one machine and time out on another, so re-executing the code on different hardware
can change a handful of verdicts. The verdicts from the analysis run on the ISCA cluster are kept in
data/results/execution/<model>.jsonl, one line per proposal:

    {"session_id": ..., "i": <proposal index>, "sha": <sha256 of task id + code>,
     "correct": true/false, "heuristic_score": float or null}

The analysis uses a recorded verdict when the session, the proposal index and the code hash all
match; anything not recorded (for example a new trace) is executed as normal. `--reexecute` on
scripts/analyze.py and scripts/rescore_all.py ignores the record and runs every proposal again.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RECORD_DIR = Path(__file__).resolve().parents[2] / "data" / "results" / "execution"
USE_RECORD = True                      # set False by --reexecute
_RECORDS: dict[tuple[str, int], dict] | None = None


def code_sha(task_id: str, code: str) -> str:
    return hashlib.sha256((task_id + "\0" + code).encode("utf-8")).hexdigest()


def _load() -> dict[tuple[str, int], dict]:
    global _RECORDS
    if _RECORDS is None:
        _RECORDS = {}
        if RECORD_DIR.is_dir():
            for f in sorted(RECORD_DIR.glob("*.jsonl")):
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            r = json.loads(line)
                            _RECORDS[(r["session_id"], int(r["i"]))] = r
    return _RECORDS


def lookup(session_id: str, i: int, task_id: str, code: str) -> dict | None:
    """The recorded verdict for proposal i of this session, or None (then the code is executed)."""
    if not USE_RECORD:
        return None
    r = _load().get((session_id, i))
    if r is None or r.get("sha") != code_sha(task_id, code):
        return None
    return r
