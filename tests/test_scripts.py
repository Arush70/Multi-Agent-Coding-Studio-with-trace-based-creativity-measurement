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


def test_tau_diagnostics_checks_only_where_the_facilitator_can_act():
    import numpy as np
    from tau_diagnostics import window_checks
    X = np.random.default_rng(0).random((5, 26))
    assert len(window_checks(X, k=5, window=3)) == 2              # after proposals 3 and 4, not 5


def test_resolve_tasks_accepts_a_long_list_of_ids():
    from build_reference import resolve_tasks
    from macs.tasks.bank import load_bank
    ids = [t.id for t in load_bank()]
    assert [t.id for t in resolve_tasks(",".join(ids))] == ids


def test_model_consistency_reads_per_model_summaries(tmp_path):
    import json as _json
    import random
    import subprocess as sp
    import sys as _sys
    from pathlib import Path as _P
    rng = random.Random(0)
    for model, shift in (("m-a", 0.0), ("m-b", 0.1)):
        sessions, per_task = [], {}
        for t in range(6):
            fam = {}
            for c, mu in (("A", 0.1), ("B", 0.5), ("C", 0.1), ("D", 0.6), ("D_minus", 0.55)):
                for s in range(3):
                    sessions.append({"task_id": f"t{t}", "condition": c, "n_proposals": 5, "n_correct": 4,
                                     "correct_and_novel_rate": mu + shift + rng.gauss(0, 0.05),
                                     "mean_novelty": mu * 3 + rng.gauss(0, 0.1), "tokens": 1000 + rng.random()})
                fam[c] = 1 + 3 * mu + rng.gauss(0, 0.1)
            per_task[f"t{t}"] = {"families_at_m": fam}
        d = tmp_path / "models" / model
        d.mkdir(parents=True)
        (d / "summary.json").write_text(_json.dumps({"model": model, "sessions": sessions,
                                                      "pooled_diversity": {"per_task": per_task}}))
    root = _P(__file__).resolve().parents[1]
    out = tmp_path / "mc.json"
    r = sp.run([_sys.executable, str(root / "scripts" / "model_consistency.py"), "--root", str(tmp_path / "models"),
                "--out", str(out)], capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    rep = _json.loads(out.read_text())
    meta = rep["contrasts"]["correct_and_novel_rate: B - A"]["meta"]
    assert rep["models"] == ["m-a", "m-b"] and meta["k"] == 2 and meta["same_sign"] == 2
    assert abs(meta["random"] - 0.4) < 0.05


def test_frontier_report(tmp_path):
    import json as _json, subprocess as sp, sys as _sys
    from pathlib import Path as _P
    sessions = []
    for c, pr, nz in (("A", 1.0, 0.0), ("D", 0.8, 2.0), ("S", 0.7, 0.5)):
        sessions.append({"task_id": "t", "condition": c, "scoring": "tests",
                         "proposals": [{"correct": i < pr * 10, "novelty": nz, "heuristic_score": None} for i in range(10)]})
        sessions.append({"task_id": "h", "condition": c, "scoring": "heuristic",
                         "proposals": [{"correct": True, "novelty": float(i), "heuristic_score": 0.9 + 0.05 * i} for i in range(5)]})
    (tmp_path / "s.json").write_text(_json.dumps({"sessions": sessions}))
    root = _P(__file__).resolve().parents[1]
    r = sp.run([_sys.executable, str(root / "scripts" / "frontier.py"), "--summary", str(tmp_path / "s.json"),
                "--out", str(tmp_path / "f.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    f = _json.loads((tmp_path / "f.json").read_text())
    assert f["frontier"]["A"]["pareto"] and f["frontier"]["D"]["pareto"] and not f["frontier"]["S"]["pareto"]
    assert abs(f["tier3"]["D"]["rho_novelty_score"] - 1.0) < 1e-9 and f["tier3"]["D"]["best"] == 1.1
