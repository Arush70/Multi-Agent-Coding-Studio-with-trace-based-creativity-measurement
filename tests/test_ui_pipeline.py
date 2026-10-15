"""Dashboard back end: background jobs, reference status and archiving, run plans, SC3 labelling."""
from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace

import pytest

from macs.ui import jobs, pipeline


def _wait(kind, d, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = jobs.status(kind, d)
        if st["state"] != "running":
            return st
        time.sleep(0.2)
    return jobs.status(kind, d)


@pytest.mark.skipif(sys.platform == "win32", reason="process-group handling differs; CI is Linux")
def test_job_runs_in_background_records_exit_code_and_progress(tmp_path):
    script = tmp_path / "count.py"
    script.write_text("import sys\nfor i in range(1, 4):\n    print(f'[{i}/3] step')\n"
                      "sys.exit(int(sys.argv[1]))\n", encoding="utf-8")
    d = tmp_path / "jobs"
    jobs.start("analyze", str(script), ["0"], jobs_dir=d)
    st = _wait("analyze", d)
    assert st["state"] == "finished" and st["returncode"] == 0
    assert st["progress"] == (3, 3) and "[3/3] step" in st["log"]
    jobs.start("analyze", str(script), ["2"], jobs_dir=d)
    assert _wait("analyze", d)["state"] == "failed"


@pytest.mark.skipif(sys.platform == "win32", reason="process-group handling differs; CI is Linux")
def test_job_can_be_stopped_and_blocks_a_second_model_job(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\nprint('[1/9] start', flush=True)\ntime.sleep(60)\n", encoding="utf-8")
    d = tmp_path / "jobs"
    jobs.start("multirun", str(script), [], jobs_dir=d)
    time.sleep(1.0)
    assert jobs.status("multirun", d)["state"] == "running"
    with pytest.raises(RuntimeError):
        jobs.start("references", str(script), [], jobs_dir=d)      # same model server
    assert jobs.stop("multirun", d) is True
    time.sleep(0.5)
    assert jobs.status("multirun", d)["state"] == "stopped"
    assert jobs.stop("multirun", d) is False


def test_parse_progress():
    assert jobs.parse_progress("[1/45] a\n[12/45] b") == (12, 45)
    assert jobs.parse_progress("nothing") is None


def test_reference_table_and_archive(tmp_path):
    root, arch = tmp_path / "ref", tmp_path / "arch"
    root.mkdir()
    tasks = [SimpleNamespace(id=t, tier=1) for t in ("ta", "tb", "tc")]
    (root / "ta.jsonl").write_text("".join(json.dumps({"seed": i, "model": "m9", "revision": "r"}) + "\n"
                                           for i in range(100)), encoding="utf-8")
    (root / "tb.jsonl").write_text(json.dumps({"seed": 0, "model": "m7", "revision": "x"}) + "\n", encoding="utf-8")
    rows = {r["task"]: r for r in pipeline.reference_table(tasks, root, model="m9")}
    assert rows["ta"]["state"] == "ready" and rows["tb"]["state"] == "other model"
    assert rows["tc"]["state"] == "missing"
    dest, n = pipeline.archive_references(["tb"], root, arch, stamp="s")
    assert n == 1 and (dest / "tb.jsonl").exists() and (root / "ta.jsonl").exists()
    dest2, n2 = pipeline.archive_references(None, root, arch, stamp="s")
    assert n2 == 1 and dest2.name == "s-1" and not list(root.glob("*.jsonl"))


def test_seeds_spec():
    assert pipeline.seeds_spec(1, 6) == "1-6" and pipeline.seeds_spec(3, 1) == "3"
    with pytest.raises(ValueError):
        pipeline.seeds_spec(1, 0)


def test_list_profiles_hides_keys(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text("default_profile: local\nprofiles:\n  local:\n    base_url: http://x/v1\n"
                   "    api_key: SECRET\n    model: qwen3.5:9b\n    revision: abc\n", encoding="utf-8")
    p = pipeline.list_profiles(cfg)
    assert p[0]["name"] == "local" and p[0]["default"] and "SECRET" not in json.dumps(p)


def _trace(path, task, cond, seed, model, impls):
    from macs.orchestrator import ORCHESTRATOR_VERSION
    from macs.trace.writer import TraceWriter
    header = {"model": model, "revision": "r", "quantisation": "q", "temperature": 0.7, "seed": seed,
              "prompt_hashes": {"x": "y"}, "orchestrator_version": ORCHESTRATOR_VERSION}
    with TraceWriter(path, session_id=path.stem, task_id=task, condition=cond, header=header) as tr:
        for i, code in enumerate(impls):
            tr.log("engineer", "implementation", content=code, proposal_id=f"p{i}")
        tr.end_note = "completed"


def test_run_plan_counts_done_cells_for_the_same_model_only(tmp_path):
    _trace(tmp_path / "a.jsonl", "t1", "A", 1, "m9", ["x=1"])
    _trace(tmp_path / "b.jsonl", "t1", "B", 1, "m7", ["x=1"])
    plan = pipeline.run_plan(["t1"], ["A", "B"], [1, 2], "m9", traces_dir=tmp_path)
    assert plan == {"total": 4, "done": 1, "to_run": 3}


def test_label_pool_save_and_agreement(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    _trace(traces / "s1.jsonl", "t1", "B", 1, "m9", ["a=1", "b=2", "c=3"])
    _trace(traces / "s2.jsonl", "t1", "A", 1, "m9", ["d=4"])
    pool = pipeline.label_pool(traces, per_task=3)
    assert len(pool) == 3 and all("condition" not in r for r in pool)
    assert pool == pipeline.label_pool(traces, per_task=3)            # fixed order
    p1, p2 = pipeline.labels_path("rater1", tmp_path), pipeline.labels_path("rater2", tmp_path)
    assert p1.name == "labels.jsonl"
    for item, (f1, f2) in zip(pool, [("set", "set"), ("sort", "sort"), ("set", "sort")]):
        pipeline.save_label(p1, item, f1)
        pipeline.save_label(p2, item, f2)
    pipeline.save_label(p1, pool[0], "counter")                          # a change overwrites
    assert len(pipeline.load_labels(p1)) == 3
    ag = pipeline.rater_agreement(p1, p2)
    assert ag["n"] == 3 and ag["agreement"] == pytest.approx(1 / 3, abs=1e-3)
