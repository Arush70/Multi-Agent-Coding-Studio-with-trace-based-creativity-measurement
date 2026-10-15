"""S8 report: planned contrasts with 95% CIs, Holm-adjusted p-values, and the minimum detectable
effect for the main-run design. Reads data/results/summary.json written by scripts/analyze.py.

Primary outcomes and the Holm family (PREREGISTRATION, SC1, SC2, SC4):
  * SC1 diversity       families at a matched number of solutions, per task and condition
                        (contrasts B - A, D - C)
  * SC2 creativity      correct-and-novel rate per session (B - A, D - C)
  * SC4 Facilitator     families at matched m (D - D_minus)
Secondary (exploratory, unadjusted): mean novelty z, pass rate, tokens, each with all three
contrasts. Cost is reported beside every gain, as the proposal requires.

    python scripts/analyze.py                     # first
    python scripts/stats_report.py                # then this
    python scripts/stats_report.py --tasks 21 --seeds 6   # MDE for that design
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macs.analysis.stats import (  # noqa: E402
    factorial_fit, facilitator_fit, holm, mde, planned_contrasts)

PRIMARY = [("families_matched", "B - A"), ("families_matched", "D - C"),
           ("correct_and_novel_rate", "B - A"), ("correct_and_novel_rate", "D - C"),
           ("families_matched", "D - D_minus")]
SECONDARY = ["mean_novelty", "pass_rate", "tokens"]


def session_rows(summary: dict) -> list[dict]:
    rows = []
    for s in summary.get("sessions", []):
        r = dict(s)
        n = s.get("n_proposals") or 0
        r["pass_rate"] = (s.get("n_correct", 0) / n) if n else None
        rows.append(r)
    return rows


def cell_rows(summary: dict) -> list[dict]:
    """One row per task and condition: families at the matched number of solutions."""
    per_task = (summary.get("pooled_diversity") or {}).get("per_task", {})
    return [{"task_id": t, "condition": c, "families_matched": v}
            for t, d in per_task.items() for c, v in d.get("families_at_m", {}).items()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="data/results/summary.json")
    ap.add_argument("--tasks", type=int, default=21, help="tasks in the planned main run (for the MDE)")
    ap.add_argument("--seeds", type=int, default=6, help="seeds per cell in the planned main run")
    ap.add_argument("--out", default="data/results/stats.json")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    sessions, cells = session_rows(summary), cell_rows(summary)
    data = {"families_matched": cells}
    for o in ["correct_and_novel_rate", *SECONDARY]:
        data[o] = sessions

    contrasts = {}
    for outcome, rows in data.items():
        for c in planned_contrasts(rows, outcome):
            contrasts[c.label] = c
    primary = [contrasts.get(f"{o}: {k}") for o, k in PRIMARY]
    adj = holm([c.p if c else float("nan") for c in primary])
    for c, a in zip(primary, adj):
        if c is not None:
            c.p_holm = a

    n_tasks = len({r["task_id"] for r in sessions})
    print(f"S8 planned contrasts  ({n_tasks} task(s); {len(sessions)} sessions; "
          f"model: outcome ~ studio * two_learners + (1 | task), REML)\n")
    head = f"{'contrast':<40}{'estimate':>10}{'95% CI':>22}{'p':>9}{'p (Holm)':>10}"
    print("PRIMARY (Holm family of 5)\n" + head)
    for (o, k), c in zip(PRIMARY, primary):
        if c is None:
            print(f"{o + ': ' + k:<40}{'not estimable (missing conditions or <2 tasks)':>51}")
            continue
        print(f"{c.label:<40}{c.estimate:>10.3f}   [{c.ci_low:>7.3f}, {c.ci_high:>7.3f}]{c.p:>9.3f}{c.p_holm:>10.3f}")
    print("\nSECONDARY (exploratory, unadjusted)\n" + head)
    for o in SECONDARY:
        for k in ("B - A", "D - C", "D - D_minus"):
            c = contrasts.get(f"{o}: {k}")
            if c is not None:
                print(f"{c.label:<40}{c.estimate:>10.3f}   [{c.ci_low:>7.3f}, {c.ci_high:>7.3f}]{c.p:>9.3f}{'':>10}")

    print(f"\nMINIMUM DETECTABLE EFFECT for {args.tasks} tasks x {args.seeds} seeds per cell "
          f"(80% power, two-sided, Holm-strict alpha 0.05/5):")
    mdes = {}
    for o in ("correct_and_novel_rate", "families_matched"):
        f = factorial_fit(data[o], o)
        if f is None:
            print(f"  {o:<26} not estimable yet")
            continue
        reps = 1 if o == "families_matched" else args.seeds
        mdes[o] = {"sigma2_residual": f.sigma2, "sigma2_task": f.sigma2_task,
                   "mde": mde(f.sigma2, args.tasks, reps, n_tests=5)}
        print(f"  {o:<26} residual SD {f.sigma2 ** 0.5:.3f}, task SD {f.sigma2_task ** 0.5:.3f}"
              f"  ->  MDE {mdes[o]['mde']:.3f}")
    g = facilitator_fit(cells, "families_matched")
    if g is not None:
        mdes["families_matched (D - D_minus)"] = {"sigma2_residual": g.sigma2,
                                                  "mde": mde(g.sigma2, args.tasks, 1, n_tests=5)}
        print(f"  {'families (D - D_minus)':<26} residual SD {g.sigma2 ** 0.5:.3f}"
              f"  ->  MDE {mdes['families_matched (D - D_minus)']['mde']:.3f}")
    print("\nPilot figures are for planning only; effects are reported from the main run.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"primary": [c.as_dict() for c in primary if c],
                               "secondary": [c.as_dict() for k, c in contrasts.items()
                                             if c not in primary],
                               "mde": mdes, "design": {"tasks": args.tasks, "seeds": args.seeds}},
                              indent=1), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
