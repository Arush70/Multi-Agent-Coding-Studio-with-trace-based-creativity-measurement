"""S8 report: planned contrasts with 95% CIs, Holm-adjusted p-values, and the minimum detectable
effect for the main-run design. Reads data/results/summary.json written by scripts/analyze.py.

Primary outcomes and the Holm family (PREREGISTRATION, SC1, SC2, SC4):
  * SC1 diversity       families at a matched number of solutions, per task and condition
                        (contrasts B - A, D - C)
  * SC2 creativity      correct-and-novel rate per session (B - A, D - C)
  * SC4 Facilitator     families at matched m (D - D_minus; E - E_minus)
  * baselines           D - S (equal token budget), D - P (single diversity prompt), both outcomes
Secondary (exploratory, unadjusted): mean novelty z, pass rate, tokens, each with all three
contrasts, the interactions and the scaffold effect. Cost is reported beside every gain, as the proposal requires.

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
    factorial_fit, pairwise_fit, holm, mde, planned_contrasts)

# Holm family (amended 2026-10-20, before any main-run data). Contrasts whose conditions were not
# run are reported as not estimable and are left out of the adjustment.
PRIMARY = [("families_matched", "B - A"), ("families_matched", "D - C"),
           ("correct_and_novel_rate", "B - A"), ("correct_and_novel_rate", "D - C"),
           ("families_matched", "D - D_minus"), ("families_matched", "E - E_minus"),
           ("families_matched", "D - S"), ("correct_and_novel_rate", "D - S"),
           ("families_matched", "D - P"), ("correct_and_novel_rate", "D - P")]
SECONDARY = ["mean_novelty", "pass_rate", "tokens", "learner_uptake_final", "learner_led_share"]


def _task_sets() -> dict[str, str]:
    from macs.tasks.bank import load_bank
    return {t.id: t.task_set for t in load_bank()}


def in_set(task_id: str, task_set: str, sets: dict | None = None) -> bool:
    """main = the pre-registered task bank (SC1); contamination_check = the EvoEval rewrites;
    all = both."""
    if task_set == "all":
        return True
    sets = sets if sets is not None else _task_sets()
    return sets.get(task_id, "main") == task_set


def session_rows(summary: dict, task_set: str = "main") -> list[dict]:
    rows = []
    sets = _task_sets()
    for s in summary.get("sessions", []):
        if not in_set(s.get("task_id", ""), task_set, sets):
            continue
        r = dict(s)
        n = s.get("n_proposals") or 0
        r["pass_rate"] = (s.get("n_correct", 0) / n) if n else None
        rows.append(r)
    return rows


def cell_rows(summary: dict, task_set: str = "main") -> list[dict]:
    """One row per task and condition: families at the matched number of solutions."""
    per_task = (summary.get("pooled_diversity") or {}).get("per_task", {})
    sets = _task_sets()
    return [{"task_id": t, "condition": c, "families_matched": v}
            for t, d in per_task.items() if in_set(t, task_set, sets)
            for c, v in d.get("families_at_m", {}).items()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="data/results/summary.json")
    ap.add_argument("--tasks", type=int, default=21, help="tasks in the planned main run (for the MDE)")
    ap.add_argument("--seeds", type=int, default=6, help="seeds per cell in the planned main run")
    ap.add_argument("--out", default="data/results/stats.json")
    ap.add_argument("--task-set", default="main", choices=["main", "contamination_check", "all"],
                    help="main = the 24-task bank (pre-registered analysis); contamination_check = the EvoEval "
                         "rewrites, analysed separately")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    sessions, cells = session_rows(summary, args.task_set), cell_rows(summary, args.task_set)
    print(f"task set: {args.task_set}")
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
    n_est = sum(1 for c in primary if c is not None)
    print(f"PRIMARY (Holm family: {n_est} estimable of {len(PRIMARY)})\n" + head)
    for (o, k), c in zip(PRIMARY, primary):
        if c is None:
            print(f"{o + ': ' + k:<40}{'not estimable (missing conditions or <2 tasks)':>51}")
            continue
        print(f"{c.label:<40}{c.estimate:>10.3f}   [{c.ci_low:>7.3f}, {c.ci_high:>7.3f}]{c.p:>9.3f}{c.p_holm:>10.3f}")
    print("\nSECONDARY (exploratory, unadjusted)\n" + head)
    for c in contrasts.values():
        if c in primary:
            continue
        print(f"{c.label:<40}{c.estimate:>10.3f}   [{c.ci_low:>7.3f}, {c.ci_high:>7.3f}]{c.p:>9.3f}{'':>10}")

    print(f"\nMINIMUM DETECTABLE EFFECT for {args.tasks} tasks x {args.seeds} seeds per cell "
          f"(80% power, two-sided, Holm-strict alpha 0.05/{len(PRIMARY)}):")
    mdes = {}
    for o in ("correct_and_novel_rate", "families_matched"):
        f = factorial_fit(data[o], o)
        if f is None:
            print(f"  {o:<26} not estimable yet")
            continue
        reps = 1 if o == "families_matched" else args.seeds
        mdes[o] = {"sigma2_residual": f.sigma2, "sigma2_task": f.sigma2_task,
                   "mde": mde(f.sigma2, args.tasks, reps, n_tests=len(PRIMARY))}
        print(f"  {o:<26} residual SD {f.sigma2 ** 0.5:.3f}, task SD {f.sigma2_task ** 0.5:.3f}"
              f"  ->  MDE {mdes[o]['mde']:.3f}")
    for a, b in (("D", "D_minus"), ("E", "E_minus"), ("D", "S"), ("D", "P")):
        g = pairwise_fit(cells, "families_matched", a, b)
        if g is not None:
            key = f"families_matched ({a} - {b})"
            mdes[key] = {"sigma2_residual": g.sigma2, "mde": mde(g.sigma2, args.tasks, 1, n_tests=len(PRIMARY))}
            print(f"  {key:<26} residual SD {g.sigma2 ** 0.5:.3f}  ->  MDE {mdes[key]['mde']:.3f}")
    print("\nPilot figures are for planning only; effects are reported from the main run.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"primary": [c.as_dict() for c in primary if c],
                               "secondary": [c.as_dict() for k, c in contrasts.items()
                                             if c not in primary],
                               "mde": mdes, "design": {"tasks": args.tasks, "seeds": args.seeds},
                               "task_set": args.task_set},
                              indent=1), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
