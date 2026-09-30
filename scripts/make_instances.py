"""Freeze the Tier 3 test instances, once, reproducibly.

Arush - the hard/open problems are scored on fixed problem instances (item lists, distance
matrices, graphs, job lists). Those instances must be FROZEN: generated once from fixed seeds,
written next to the task, committed, and never regenerated - otherwise a re-run would score
different heuristics on different problems and nothing would be comparable. This script does that.
cap_set has no instances (it's scored on dimensions 3/4/5 directly), so it's skipped.

Run once, after any change to a task's instance size/seeds:
    python scripts/make_instances.py              # writes tasks/tier3/<task>/instances.json
    python scripts/make_instances.py --overwrite  # replace existing frozen instances

The seeds and sizes are read from each task.yaml's `heuristic:` block, so the task file is the
single source of truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from macs.tasks.bank import load_bank

# how big each instance is (kept small so the sandbox scores a heuristic in well under a second,
# but large enough that different heuristics actually separate)
SIZES = {
    "t3_bin_packing": {"n_items": 50, "lo": 0.05, "hi": 0.95},
    "t3_scheduling": {"n_jobs": 40, "lo": 1.0, "hi": 10.0},
    "t3_tsp_construction": {"n_cities": 20},
    "t3_graph_coloring": {"n_vertices": 30, "edge_prob": 0.3},
}


def _bin_packing(seed: int, cfg: dict) -> list[float]:
    rng = np.random.default_rng(seed)
    return [round(float(x), 4) for x in rng.uniform(cfg["lo"], cfg["hi"], cfg["n_items"])]


def _scheduling(seed: int, cfg: dict) -> list[float]:
    rng = np.random.default_rng(seed)
    return [round(float(x), 2) for x in rng.uniform(cfg["lo"], cfg["hi"], cfg["n_jobs"])]


def _tsp(seed: int, cfg: dict) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    n = cfg["n_cities"]
    pts = rng.uniform(0.0, 1.0, (n, 2))
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return [[round(float(x), 6) for x in row] for row in D]


def _graph(seed: int, cfg: dict) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    n = cfg["n_vertices"]
    adj: dict[str, list[int]] = {str(v): [] for v in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < cfg["edge_prob"]:
                adj[str(i)].append(j)
                adj[str(j)].append(i)
    return adj


_GENERATORS = {
    "t3_bin_packing": _bin_packing,
    "t3_scheduling": _scheduling,
    "t3_tsp_construction": _tsp,
    "t3_graph_coloring": _graph,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true", help="replace instances.json even if it already exists")
    args = ap.parse_args()

    for task in load_bank(tier=3):
        gen = _GENERATORS.get(task.id)
        inst_file = task.heuristic.get("instance_file")
        if gen is None or not inst_file:
            print(f"  {task.id:<24} no instances (skipped)")
            continue
        assert task.path is not None
        out = task.path / inst_file
        if out.exists() and not args.overwrite:
            print(f"  {task.id:<24} already frozen: {out} (use --overwrite to replace)")
            continue
        seeds = task.heuristic.get("seeds", [1, 2, 3, 4, 5])
        cfg = SIZES[task.id]
        instances = [gen(s, cfg) for s in seeds]
        out.write_text(json.dumps({
            "task_id": task.id, "seeds": list(seeds), "size": cfg, "instances": instances,
        }, indent=0))
        print(f"  {task.id:<24} wrote {len(instances)} instances -> {out}")


if __name__ == "__main__":
    main()
