"""Open-problem search loop with a fake model."""
import sys

import pytest

from macs.analysis.score import task_by_id
from macs.search.loop import read_curve, run_search
from tests.test_orchestrator import FakeCompletion, FakeProfile

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="subprocess sandbox rlimits differ on Windows")

GRID = """def pack_circles(n):
    import math
    k = math.ceil(math.sqrt(n)); r = 1 / (2 * k) * {f}
    return [((i % k) * 2 * (1 / (2 * k)) + 1 / (2 * k), (i // k) * 2 * (1 / (2 * k)) + 1 / (2 * k), r) for i in range(n)]
"""


class Fake:
    """Writes grid packings whose radius factor improves when shown earlier programs."""
    profile = FakeProfile()
    def __init__(self):
        self.calls = 0
    def chat(self, messages, *, seed=None, **kw):
        self.calls += 1
        sys_ = messages[0]["content"]
        if "Innovator" in sys_:
            return FakeCompletion(text=f"Strategy {self.calls}: shrink-and-grow grid.")
        better = "Score" in messages[1]["content"]
        f = min(0.5 + 0.05 * self.calls, 1.0) if better else 0.6
        return FakeCompletion(text="```python\n" + GRID.format(f=f) + "```")


@posix_only
@pytest.mark.parametrize("strategy", ["random", "single", "studio"])
def test_search_runs_logs_and_resumes(tmp_path, strategy):
    task = task_by_id("t3_circle_packing")
    p = run_search(task, Fake(), strategy=strategy, budget=4, seed=1, out_dir=tmp_path, log=None)
    c = read_curve(p)
    assert c["n"] == 4 and c["strategy"] == strategy and all(0 < v <= 1.0 + 1e-9 for v in c["curve"])
    assert c["curve"] == sorted(c["curve"])                     # best-so-far never decreases
    # resume: asking for a larger budget continues from step 4
    p2 = run_search(task, Fake(), strategy=strategy, budget=6, seed=1, out_dir=tmp_path, log=None)
    assert read_curve(p2)["n"] == 6
    if strategy == "single":
        assert c["best"] > 0.6                                  # feedback lets it improve


@posix_only
def test_report_compares_strategies(tmp_path):
    import json, subprocess
    from pathlib import Path
    task = task_by_id("t3_circle_packing")
    for s in ("random", "single", "studio"):
        for seed in (1, 2):
            run_search(task, Fake(), strategy=s, budget=3, seed=seed, out_dir=tmp_path / "data" / "search", log=None)
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "search_open.py"), "report"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "data" / "results" / "search_report.json").read_text())
    t = rep["tasks"]["t3_circle_packing"]
    assert t["best_known"] == 2.635983 and set(t["strategies"]) == {"random", "single", "studio"}
    assert t["comparisons"]["studio - single"]["n_seeds"] == 2
