"""Tests for the batch runner (scripts/run_experiment.py) and the dashboard headline.

Check the parts that make re-running safe: seed parsing, task selection, and that only complete
traces from the current orchestrator count as done (a crashed session is run again).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from macs.orchestrator import ORCHESTRATOR_VERSION
from macs.trace.writer import TraceWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts" / "run_experiment.py")
rx = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rx)

HEADER = {"model": "fake", "revision": "0", "quantisation": "none", "temperature": 0.7,
          "prompt_hashes": {"x": "y"}}


def _trace(path, cond, seed, version, end):
    h = dict(HEADER, seed=seed)
    if version:
        h["orchestrator_version"] = version
    with TraceWriter(path, session_id=path.stem, task_id="t1_mbpp19_test_duplicate", condition=cond, header=h) as tr:
        tr.end_note = end


def test_parse_seeds():
    assert rx.parse_seeds("1-6") == [1, 2, 3, 4, 5, 6]
    assert rx.parse_seeds("1,3, 5") == [1, 3, 5]
    assert rx.parse_seeds("2,4-5") == [2, 4, 5]


def test_done_cells_only_counts_complete_current_traces(tmp_path):
    _trace(tmp_path / "a.jsonl", "A", 1, ORCHESTRATOR_VERSION, "completed; tokens_used=10")
    _trace(tmp_path / "b.jsonl", "B", 1, ORCHESTRATOR_VERSION, "stopped: token ceiling")
    _trace(tmp_path / "c.jsonl", "C", 1, ORCHESTRATOR_VERSION, None)        # crashed
    _trace(tmp_path / "d.jsonl", "D", 1, None, "completed")                  # old orchestrator
    done = rx.done_cells(tmp_path)
    assert done == {("t1_mbpp19_test_duplicate", "A", 1), ("t1_mbpp19_test_duplicate", "B", 1)}


def test_resolve_tasks_accepts_ids_and_tiers():
    assert [t.id for t in rx.resolve_tasks("t1_mbpp19_test_duplicate,t3_bin_packing")] == \
        ["t1_mbpp19_test_duplicate", "t3_bin_packing"]
    assert all(t.tier == 3 for t in rx.resolve_tasks("tier3")) and len(rx.resolve_tasks("tier3")) == 5


def test_dashboard_headline_uses_matched_families():
    from macs.ui.app import studio_vs_tutor
    summary = {"A": {"avg_families": 1, "avg_families_matched": 2.0},
               "B": {"avg_families": 6, "avg_families_matched": 3.0}}
    sv = studio_vs_tutor(summary)
    assert sv == {"studio_families": 3.0, "tutor_families": 2.0, "matched": True}
    assert studio_vs_tutor({"A": {"avg_families": 1}, "B": {"avg_families": 6}})["matched"] is False


def test_archive_traces_moves_files_and_never_overwrites(tmp_path):
    from macs.ui.app import archive_traces
    traces, archive = tmp_path / "traces", tmp_path / "archive"
    traces.mkdir()
    assert archive_traces(traces, archive, stamp="s") == (None, 0)
    for name in ("a.jsonl", "b.jsonl"):
        (traces / name).write_text("{}\n")
    (traces / "keep.txt").write_text("x")
    dest, n = archive_traces(traces, archive, stamp="s")
    assert n == 2 and dest == archive / "s"
    assert sorted(p.name for p in dest.iterdir()) == ["a.jsonl", "b.jsonl"]
    assert [p.name for p in traces.iterdir()] == ["keep.txt"]
    (traces / "c.jsonl").write_text("{}\n")
    dest2, n2 = archive_traces(traces, archive, stamp="s")
    assert n2 == 1 and dest2 == archive / "s-1"
    assert (archive / "s" / "a.jsonl").exists()


def test_reference_refuses_to_mix_models(tmp_path):
    import json
    import pytest
    from build_reference import build_task_reference

    class _Prof:
        model, revision = "qwen3.5:9b", "b"

    class _Client:
        profile = _Prof()

    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"seed": 100000, "code": "x=1", "model": "qwen2.5-coder:7b",
                                "revision": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="different model"):
        build_task_reference(object(), _Client(), n=2, path=path)
