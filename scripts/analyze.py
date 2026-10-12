"""Score every trace and print the per-condition results table.

Usage:

    python scripts/analyze.py --sandbox subprocess

Each session is scored on the held-out tests (Tier 1/2) or the frozen instances (Tier 3
heuristic tasks). Diversity, correctness and (where a reference set exists) novelty are
computed, aggregated by condition, printed as a table and saved to
data/results/summary.json for the dashboard. The `heur` column is the mean Tier 3 score
relative to the baseline heuristic (>1 beats it); it is blank for conditions with no
Tier 3 sessions. Traces from older orchestrator versions and crashed sessions are skipped
unless --include-old is given.
"""
import argparse, json
from pathlib import Path

from macs.analysis.aggregate import analyse_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/traces")
    ap.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"])
    ap.add_argument("--out", default="data/results/summary.json")
    ap.add_argument("--include-old", action="store_true",
                    help="also score traces from older orchestrator versions / crashed sessions (not comparable)")
    args = ap.parse_args()

    # load any reference sets that exist, so novelty is computed where possible
    references = {}
    try:
        from macs.metrics.reference import build_reference_stats, REFERENCE_ROOT, REFERENCE_N
        small = []
        for f in Path(REFERENCE_ROOT).glob("*.jsonl"):
            tid = f.stem
            try:
                references[tid] = build_reference_stats(tid)
                if len(references[tid].X_ref) < REFERENCE_N:
                    small.append(f"{tid} ({len(references[tid].X_ref)})")
            except Exception:
                pass
        if small:
            print(f"[warn] reference sets below {REFERENCE_N} parseable samples (novelty less reliable): "
                  + ", ".join(sorted(small)) + "\n       top up with: python scripts/build_reference.py --tasks all")
    except Exception:
        pass

    sessions, summary, pooled = analyse_dir(args.traces, backend=args.sandbox, references=references,
                                            return_pooled=True, include_legacy=args.include_old)
    print(f"scored {len(sessions)} sessions; novelty available for {len(references)} tasks\n")
    if not sessions:
        print("no comparable traces yet - run sessions with the current code first, e.g.\n"
              "  python scripts/run_experiment.py --tasks t1_mbpp19_test_duplicate --seeds 1-3\n")
    hdr = (f"{'cond':<9}{'sessions':<9}{'families':<10}{'fam@10':<8}{'correct':<9}"
           f"{'novelty':<9}{'corr&novel':<12}{'heur':<8}{'tokens':<9}{'interv'}")
    print(hdr); print("-" * len(hdr))
    for cond, m in summary.items():
        print(f"{cond:<9}{m['n_sessions']:<9}{str(m['avg_families']):<10}{str(m['avg_families_first10']):<8}"
              f"{str(m['avg_correct']):<9}{str(m['avg_mean_novelty']):<9}{str(m['avg_correct_and_novel']):<12}"
              f"{str(m.get('avg_heuristic_score')):<8}{str(m.get('avg_tokens')):<9}{str(m.get('avg_interventions'))}")

    # Families at a MATCHED number of solutions per task (all conditions clustered together).
    # This is the fair tutor-vs-studio diversity comparison; the per-session 'families' column
    # above is capped at 1 for tutors by construction.
    print("\nfamilies at matched sample size (per task, common space):")
    for tid, t in pooled["per_task"].items():
        cells = "  ".join(f"{c}={v}" for c, v in t["families_at_m"].items())
        print(f"  {tid:<40} m={t['m']:<3} {cells}")
    for cond, v in pooled["by_condition"].items():
        print(f"  {cond:<9} mean over {v['n_tasks']} task(s): {v['avg_families_at_matched_m']}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sessions": sessions, "summary": summary, "pooled_diversity": pooled}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
