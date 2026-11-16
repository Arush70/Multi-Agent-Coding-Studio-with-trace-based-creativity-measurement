"""Rescore every stored trace and rebuild every result, on any computer (no GPU, no model server).

The same steps as isca/analyze_all.sbatch, in Python so it also runs on Windows and from the
dashboard (Overview > "Rescore everything"):

  1. scripts/analyze.py for every model that has traces (in parallel, one process per model)
     -> data/results/models/<model>/summary.json
  2. scripts/stats_report.py per model (6 seeds for Qwen3.5-27B and 9B, 3 for the others)
  3. the main model (Qwen3.5-27B) copied to data/results/summary.json, its statistics,
     the contamination-check tasks, the quality-diversity frontier, the mutation test
  4. scripts/model_consistency.py (meta-analysis across models) and the open-problem report
  5. the UniXcoder check, if torch and transformers are installed (pip install -e ".[embed]")

    python scripts/rescore_all.py                  # everything
    python scripts/rescore_all.py --workers 2      # fewer parallel model analyses (less memory)

The stored traces and reference sets are read, never changed; data/results/ is rewritten.
"""
from __future__ import annotations

import argparse, importlib.util, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAIN = "qwen3.5:27b"
FULL_DESIGN = {"qwen3.5:27b", "qwen3.5:9b"}


def run(args, log=None):
    cmd = [sys.executable, *args]
    with (open(log, "w", encoding="utf-8") if log else open(os.devnull, "w")) as fh:
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh if log else None, stderr=subprocess.STDOUT if log else None,
                           env={**os.environ, "PYTHONPATH": str(ROOT)})
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"])
    ap.add_argument("--skip-mutation", action="store_true", help="leave out the mutation test (slowest extra)")
    a = ap.parse_args()
    from macs.analysis.aggregate import trace_models
    from macs.metrics.reference import model_slug
    t0 = time.time()
    res = ROOT / "data" / "results"
    (res / "models").mkdir(parents=True, exist_ok=True)
    models = sorted({m.split("@")[0] for m in trace_models(str(ROOT / "data" / "traces"))})
    steps = len(models) * 2 + 7
    done = 0

    def step(msg):
        nonlocal done
        done += 1
        print(f"[{done}/{steps}] {msg}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    print(f"models with traces: {', '.join(models)}; {a.workers} in parallel", flush=True)

    def analyse(m):
        s = model_slug(m); d = res / "models" / s; d.mkdir(parents=True, exist_ok=True)
        rc = run(["scripts/analyze.py", "--model", m, "--sandbox", a.sandbox, "--out", str(d / "summary.json")],
                 log=d / "analyze.log")
        return m, rc

    with ThreadPoolExecutor(a.workers) as ex:
        for m, rc in ex.map(analyse, models):
            step(f"analysed {m}" + ("" if rc == 0 else f"  FAILED (see data/results/models/{model_slug(m)}/analyze.log)"))
    for m in models:
        s = model_slug(m)
        rc = run(["scripts/stats_report.py", "--summary", f"data/results/models/{s}/summary.json", "--tasks", "24",
                  "--seeds", "6" if m in FULL_DESIGN else "3", "--out", f"data/results/models/{s}/stats.json"],
                 log=res / "models" / s / "stats.log")
        step(f"statistics {m}" + ("" if rc == 0 else "  FAILED"))
    shutil.copy(res / "models" / model_slug(MAIN) / "summary.json", res / "summary.json")
    run(["scripts/stats_report.py", "--tasks", "24", "--seeds", "6"], log=res / "stats.log"); step("main model statistics")
    run(["scripts/stats_report.py", "--task-set", "contamination_check", "--out", "data/results/stats_contamination.json"],
        log=res / "stats_contamination.log"); step("contamination-check tasks")
    plot = ["--plot"] if importlib.util.find_spec("matplotlib") else []
    run(["scripts/frontier.py", *plot], log=res / "frontier.log"); step("quality-diversity frontier")
    if a.skip_mutation:
        step("mutation test skipped")
    else:
        run(["scripts/mutation_test.py", "--model", MAIN, "--sandbox", a.sandbox], log=res / "mutation.log")
        step("mutation test")
    run(["scripts/model_consistency.py"], log=res / "model_consistency.log"); step("meta-analysis across models")
    run(["scripts/search_open.py", "report"], log=res / "search_report.log"); step("open-problem report")
    if importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"):
        run(["scripts/analyze.py", "--model", MAIN, "--representation", "unixcoder", "--sandbox", a.sandbox,
             "--out", "data/results/summary_unixcoder.json"], log=res / "analyze_unixcoder.log")
        step("UniXcoder check")
    else:
        step("UniXcoder check skipped (install torch and transformers: pip install -e \".[embed]\")")
    print(f"all done in {(time.time() - t0) / 60:.1f} min; results in data/results/", flush=True)


if __name__ == "__main__":
    main()
