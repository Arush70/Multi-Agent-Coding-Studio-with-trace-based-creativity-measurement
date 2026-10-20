"""Does creativity pay off? The quality-diversity frontier and the Tier 3 payoff.

Reads data/results/summary.json (scripts/analyze.py) and reports, per condition:

  * the frontier position: mean correctness (pass rate) against mean novelty z, and which
    conditions are Pareto-efficient (no other condition is both more correct and more novel);
  * on Tier 3 (hard and open problems): the best and mean heuristic score (1.0 = the standard
    baseline heuristic), and Spearman's rho between a proposal's novelty and its score, pooled
    within task: do more novel heuristics perform better?

    python scripts/frontier.py [--summary data/results/summary.json] [--plot]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="data/results/summary.json")
    ap.add_argument("--out", default="data/results/frontier.json")
    ap.add_argument("--plot", action="store_true", help="also write data/results/frontier.png (needs matplotlib)")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    by = defaultdict(list)
    for r in s.get("sessions", []):
        by[r["condition"]].extend([dict(p, task_id=r["task_id"], scoring=r.get("scoring")) for p in r.get("proposals", [])])
    pts = {}
    for c, ps in sorted(by.items()):
        corr = [p["correct"] for p in ps if p["correct"] is not None and p.get("scoring") == "tests"]
        nov = [p["novelty"] for p in ps if p["novelty"] is not None]
        pts[c] = {"pass_rate": float(np.mean(corr)) if corr else None, "novelty": float(np.mean(nov)) if nov else None,
                  "n": len(ps)}
    ok = {c: v for c, v in pts.items() if v["pass_rate"] is not None and v["novelty"] is not None}
    for c, v in ok.items():
        v["pareto"] = not any(o["pass_rate"] >= v["pass_rate"] and o["novelty"] >= v["novelty"]
                              and (o["pass_rate"], o["novelty"]) != (v["pass_rate"], v["novelty"])
                              for k, o in ok.items() if k != c)
    print("Quality-diversity frontier (Tier 1/2 proposals)")
    print(f"  {'cond':<9}{'pass rate':>10}{'novelty z':>11}{'pareto':>8}{'n':>6}")
    for c, v in ok.items():
        print(f"  {c:<9}{v['pass_rate']:>10.3f}{v['novelty']:>11.3f}{('yes' if v['pareto'] else ''):>8}{v['n']:>6}")

    # Tier 3 payoff
    tier3 = {}
    for c, ps in sorted(by.items()):
        h = [p for p in ps if p.get("scoring") == "heuristic" and p["heuristic_score"] is not None]
        if not h:
            continue
        rho_parts = []
        for tid in {p["task_id"] for p in h}:
            pts_t = [(p["novelty"], p["heuristic_score"]) for p in h if p["task_id"] == tid and p["novelty"] is not None]
            if len(pts_t) >= 4:
                r = stats.spearmanr([x for x, _ in pts_t], [y for _, y in pts_t])[0]
                if np.isfinite(r):
                    rho_parts.append(r)
        tier3[c] = {"best": float(max(p["heuristic_score"] for p in h)),
                    "mean": float(np.mean([p["heuristic_score"] for p in h])),
                    "beats_baseline": float(np.mean([p["heuristic_score"] > 1.0 for p in h])),
                    "rho_novelty_score": float(np.mean(rho_parts)) if rho_parts else None, "n": len(h)}
    if tier3:
        print("\nTier 3 payoff (score 1.0 = standard baseline heuristic)")
        print(f"  {'cond':<9}{'best':>8}{'mean':>8}{'> base':>8}{'rho(nov,score)':>16}{'n':>6}")
        for c, v in tier3.items():
            rho = "-" if v["rho_novelty_score"] is None else f"{v['rho_novelty_score']:.3f}"
            print(f"  {c:<9}{v['best']:>8.3f}{v['mean']:>8.3f}{v['beats_baseline']:>8.2f}{rho:>16}{v['n']:>6}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"frontier": pts, "tier3": tier3}, indent=1), encoding="utf-8")
    print(f"\nwritten: {out}")
    if a.plot and ok:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for c, v in ok.items():
            ax.scatter(v["novelty"], v["pass_rate"], s=40, color="#0f8b8d" if v["pareto"] else "#9aa3ad")
            ax.annotate(c, (v["novelty"], v["pass_rate"]), textcoords="offset points", xytext=(5, 4), fontsize=9)
        ax.set_xlabel("mean novelty (z)"); ax.set_ylabel("pass rate (held-out tests)")
        ax.set_title("Quality-diversity frontier"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=160)
        print(f"plot: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
