"""Reference-set loading and fitting tests; no model needed.

Check that saved solutions become a working novelty yardstick, that the analysis uses the first
n samples by seed, and that the reference builder tops up a set without rewriting it.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from macs.metrics.reference import load_reference_solutions, build_reference_stats

# a handful of ordinary, varied solutions to a toy task
SOLUTIONS = [
    "def f(n):\n    return sum(range(n+1))",
    "def f(n):\n    s = 0\n    for i in range(n+1):\n        s += i\n    return s",
    "def f(n):\n    return n*(n+1)//2",
    "def f(n):\n    total = 0\n    i = 1\n    while i <= n:\n        total += i\n        i += 1\n    return total",
    "def f(n):\n    return sum(i for i in range(1, n+1))",
    "def f(n):\n    acc = 0\n    for k in range(1, n+1):\n        acc = acc + k\n    return acc",
]


def _write(tmp_path, task_id="t_demo"):
    p = tmp_path / f"{task_id}.jsonl"
    p.write_text("\n".join(json.dumps({"task_id": task_id, "seed": i, "code": c})
                           for i, c in enumerate(SOLUTIONS)), encoding="utf-8")
    return tmp_path


def test_load_reads_all_solutions(tmp_path):
    root = _write(tmp_path)
    assert len(load_reference_solutions("t_demo", root=root)) == len(SOLUTIONS)


def test_build_stats_gives_working_yardstick(tmp_path):
    root = _write(tmp_path)
    rs = build_reference_stats("t_demo", representation="ast", k=3, root=root)
    assert rs.X_ref.shape[0] == len(SOLUTIONS)          # one row per solution
    # a solution from the same family scores near-typical; a wildly different one scores higher
    from macs.metrics.represent import ast_feature_matrix, normalise
    typical = ast_feature_matrix([normalise("def f(n):\n    return sum(range(n+1))")])
    weird = ast_feature_matrix([normalise(
        "def f(n):\n    import math\n    return int(math.pow(n,2)/2 + n/2) if n%2==0 else (n*n+n)//2")])
    assert rs.novelty(weird)[0] >= rs.novelty(typical)[0]


def test_missing_reference_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reference_solutions("does_not_exist", root=tmp_path)


# --- 100-sample references: top-up and the first-n rule --------------------------------------

def test_analysis_uses_first_n_by_seed(tmp_path):
    import json
    from macs.metrics.reference import load_reference_solutions
    root = tmp_path; p = root / "t_n.jsonl"
    rows = [{"seed": 100_000 + i, "code": f"def f(a):\n    return {i}\n"} for i in (3, 0, 2, 1)]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert load_reference_solutions("t_n", root=root, n=2) == ["def f(a):\n    return 0\n", "def f(a):\n    return 1\n"]
    assert len(load_reference_solutions("t_n", root=root, n=None)) == 4


def test_build_reference_tops_up_without_rewriting(tmp_path):
    import importlib.util, json, sys
    from pathlib import Path
    from macs.tasks.bank import load_task, TASKS_ROOT
    spec = importlib.util.spec_from_file_location("build_reference", Path(__file__).parents[1] / "scripts" / "build_reference.py")
    br = importlib.util.module_from_spec(spec); spec.loader.exec_module(br)

    class C:
        text: str
        def __init__(self, t): self.text = t; self.prompt_tokens = 1; self.completion_tokens = 1
    class Prof:
        model = "fake"; revision = "0"
    class Client:
        profile = Prof()
        def __init__(self): self.seeds = []
        def chat(self, messages, *, temperature=0.7, seed=None, **kw):
            self.seeds.append(seed)
            return C(f"```python\ndef test_duplicate(a):\n    return len(a) != len(set(a))  # {seed}\n```")

    task = load_task(TASKS_ROOT / "tier1" / "t1_mbpp19_test_duplicate")
    path = tmp_path / f"{task.id}.jsonl"
    c = Client()
    assert br.build_task_reference(task, c, n=3, path=path, log=lambda *a: None) == 3
    first = path.read_text()
    path.write_text(first + '{"task_id": "x", "seed": 1000')      # simulate an interrupted run
    c2 = Client()
    assert br.build_task_reference(task, c2, n=5, path=path, log=lambda *a: None) == 2
    assert c2.seeds == [br.REF_SEED_BASE + 3, br.REF_SEED_BASE + 4]  # only the missing seeds
    assert path.read_text().startswith(first)                       # existing samples untouched
    assert [json.loads(l)["seed"] for l in path.read_text().splitlines()] == [br.REF_SEED_BASE + i for i in range(5)]
    assert br.build_task_reference(task, Client(), n=5, path=path, log=lambda *a: None) == 0


def test_each_model_has_its_own_reference_location(tmp_path, monkeypatch):
    import json as _json
    from macs.metrics import reference as R
    monkeypatch.setattr(R, "REFERENCE_ROOT", tmp_path)
    (tmp_path / "t1.jsonl").write_text(_json.dumps({"seed": 0, "model": "qwen3.5:9b", "code": "x=1"}) + "\n",
                                       encoding="utf-8")
    assert R.reference_path("t1", "qwen3.5:9b") == tmp_path / "t1.jsonl"          # first model: legacy file
    assert R.reference_path("t1", "llama3.1:8b") == tmp_path / "models" / "llama3.1_8b" / "t1.jsonl"
    assert R.reference_path("t1") == tmp_path / "t1.jsonl"
    other = tmp_path / "models" / "llama3.1_8b"
    other.mkdir(parents=True)
    (other / "t2.jsonl").write_text(_json.dumps({"seed": 0, "model": "llama3.1:8b", "code": "y=2"}) + "\n",
                                    encoding="utf-8")
    assert R.reference_tasks("llama3.1:8b") == ["t2"]
    assert R.reference_tasks("qwen3.5:9b") == ["t1"]
    assert R.load_reference_solutions("t2", model="llama3.1:8b") == ["y=2"]
