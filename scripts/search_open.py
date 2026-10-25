"""Open-problem search study.

    python scripts/search_open.py run --tasks t3_circle_packing,t3_bin_packing --budget 40 --seeds 1-3
    python scripts/search_open.py report
    python scripts/search_open.py migrate        # once: move older flat logs into per-model folders

`run` runs every strategy (random, single, studio) with the same budget of scored programs, for
each task and seed; it can be stopped and restarted (it continues where it stopped). `report`
summarises: best score reached (1.0 = the standard baseline), best raw value against the
best-known value where one is recorded, the area under the best-so-far curve, tokens, and a
paired comparison of studio against single and random over seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_reference import resolve_tasks  # noqa: E402
from run_experiment import parse_seeds  # noqa: E402
from macs.search.loop import STRATEGIES, read_curve, run_search  # noqa: E402

OUT = Path("data/search")


def cmd_run(a):
    from macs.llm.client import LLMClient
    tasks = resolve_tasks(a.tasks)
    bad = [t.id for t in tasks if t.scoring != "heuristic"]
    if bad:
        raise SystemExit(f"search needs metric-scored (Tier 3) tasks; not: {', '.join(bad)}")
    client = LLMClient.from_config(profile=a.profile)
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    for seed in parse_seeds(a.seeds):
        for t in tasks:
            for s in strategies:
                run_search(t, client, strategy=s, budget=a.budget, seed=seed, backend=a.sandbox, out_dir=OUT)


def search_logs() -> list[Path]:
    """Every search log: data/search/<model>/*.jsonl (and any older flat data/search/*.jsonl)."""
    return sorted(OUT.glob("*/*.jsonl")) + sorted(OUT.glob("*.jsonl"))


def cmd_migrate(a):
    """Move flat data/search/*.jsonl logs (before per-model folders) into data/search/<model>/."""
    from macs.metrics.reference import model_slug
    moved = 0
    for p in sorted(OUT.glob("*.jsonl")):
        model = read_curve(p)["model"] or "unknown_model"
        dest = OUT / model_slug(model) / p.name
        if dest.exists():
            print(f"  kept {p.name}: {dest} already exists")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        p.rename(dest)
        moved += 1
    print(f"moved {moved} log(s) into per-model folders under {OUT}")


def _auc(curve: list[float]) -> float | None:
    return float(sum(curve) / len(curve)) if curve else None


def _report_tasks(runs: list[dict]) -> dict:
    import numpy as np
    from scipy import stats
    from macs.analysis.score import task_by_id
    out = {}
    for tid in sorted({r["task_id"] for r in runs}):
        task = task_by_id(tid)
        best_known = (task.heuristic or {}).get("best_known")
        rows = {}
        print(f"\n== {tid}" + (f"  (best known raw value {best_known})" if best_known else ""))
        print(f"   {'strategy':<9}{'seeds':>6}{'best score':>12}{'best raw':>11}{'AUC':>8}{'valid':>7}{'tokens':>10}{'interv':>8}")
        for s in STRATEGIES:
            rs = [r for r in runs if r["task_id"] == tid and r["strategy"] == s]
            if not rs:
                continue
            best = [r["best"] for r in rs]
            raw = [r["best_metric"] for r in rs if r["best_metric"] is not None]
            rows[s] = {"seeds": sorted(r["seed"] for r in rs), "best_mean": float(np.mean(best)),
                       "best_max": float(np.max(best)), "best_raw": max(raw) if raw else None,
                       "auc_mean": float(np.mean([_auc(r["curve"]) for r in rs])),
                       "valid_share": float(np.mean([r["valid_share"] for r in rs])),
                       "tokens_mean": float(np.mean([r["tokens"] for r in rs])),
                       "interventions": int(sum(r["interventions"] for r in rs)),
                       "by_seed": {r["seed"]: {"best": r["best"], "auc": _auc(r["curve"])} for r in rs},
                       "gap_to_best_known": (best_known - max(raw)) if (best_known and raw) else None}
            v = rows[s]
            print(f"   {s:<9}{len(rs):>6}{v['best_mean']:>12.4f}{(v['best_raw'] or float('nan')):>11.4f}"
                  f"{v['auc_mean']:>8.3f}{v['valid_share']:>7.2f}{v['tokens_mean']:>10.0f}{v['interventions']:>8}")
        comps = {}
        for other in ("single", "random"):
            if "studio" in rows and other in rows:
                seeds = sorted(set(rows["studio"]["by_seed"]) & set(rows[other]["by_seed"]))
                d = [rows["studio"]["by_seed"][k]["auc"] - rows[other]["by_seed"][k]["auc"] for k in seeds]
                test = stats.wilcoxon(d) if len(d) >= 6 and any(x != 0 for x in d) else None
                comps[f"studio - {other}"] = {"n_seeds": len(d), "mean_auc_diff": float(np.mean(d)) if d else None,
                                              "studio_better_seeds": int(sum(x > 0 for x in d)),
                                              "wilcoxon_p": float(test.pvalue) if test else None}
                c = comps[f"studio - {other}"]
                print(f"   studio vs {other}: AUC difference {c['mean_auc_diff']:+.3f}, studio better on "
                      f"{c['studio_better_seeds']}/{c['n_seeds']} seeds"
                      + (f", Wilcoxon p={c['wilcoxon_p']:.3f}" if c["wilcoxon_p"] is not None else ""))
        out[tid] = {"best_known": best_known, "strategies": rows, "comparisons": comps}
    return out


def cmd_report(a):
    runs = [read_curve(p) for p in search_logs()]
    runs = [r for r in runs if r["n"]]
    if not runs:
        raise SystemExit(f"no search logs in {OUT}; run first")
    models = sorted({r["model"] or "unknown_model" for r in runs})
    report = {"models": {}}
    for m in models:
        print(f"\n######## model {m}")
        report["models"][m] = {"tasks": _report_tasks([r for r in runs if (r["model"] or "unknown_model") == m])}
    main = a.model if a.model in report["models"] else ("qwen3.5:27b" if "qwen3.5:27b" in models else models[0])
    report["main_model"] = main
    report["tasks"] = report["models"][main]["tasks"]       # the main model's results, as before
    out = Path("data/results/search_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwritten: {out} (main model {main}; models: {', '.join(models)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tasks", default="t3_circle_packing,t3_bin_packing,t3_cap_set")
    r.add_argument("--strategies", default=",".join(STRATEGIES))
    r.add_argument("--budget", type=int, default=40, help="programs scored per run (same for every strategy)")
    r.add_argument("--seeds", default="1-3")
    r.add_argument("--profile", default="local")
    r.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"])
    rp = sub.add_parser("report")
    rp.add_argument("--model", default=None, help="model whose results go under 'tasks' (default qwen3.5:27b)")
    sub.add_parser("migrate", help="move older flat logs into per-model folders")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report, "migrate": cmd_migrate}[a.cmd](a)


if __name__ == "__main__":
    main()
