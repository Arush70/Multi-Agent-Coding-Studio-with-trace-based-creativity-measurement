"""Score every trace and print the per-condition results table. YOUR RESULTS IN ONE COMMAND.

Arush - run this after you've generated some sessions:

    python scripts/analyze.py --sandbox subprocess

It scores each session on the held-out tests (Tier 1/2) or the frozen instances (Tier 3 heuristic
tasks), computes diversity / correctness (and novelty if a reference set exists for the task),
aggregates by condition, prints a comparison table, and saves it to data/results/summary.json for
the UI to read. The `heur` column is the mean Tier 3 score vs the baseline heuristic (>1 beats it);
it is blank for conditions with no Tier 3 sessions.
"""
import argparse, json
from pathlib import Path

from macs.analysis.aggregate import analyse_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/traces")
    ap.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"])
    ap.add_argument("--out", default="data/results/summary.json")
    args = ap.parse_args()

    # load any reference sets that exist, so novelty is computed where possible
    references = {}
    try:
        from macs.metrics.reference import build_reference_stats, REFERENCE_ROOT
        for f in Path(REFERENCE_ROOT).glob("*.jsonl"):
            tid = f.stem
            try:
                references[tid] = build_reference_stats(tid)
            except Exception:
                pass
    except Exception:
        pass

    sessions, summary = analyse_dir(args.traces, backend=args.sandbox, references=references)
    print(f"scored {len(sessions)} sessions; novelty available for {len(references)} tasks\n")
    hdr = (f"{'cond':<9}{'sessions':<9}{'families':<10}{'fam@10':<8}{'correct':<9}"
           f"{'novelty':<9}{'corr&novel':<12}{'heur'}")
    print(hdr); print("-" * len(hdr))
    for cond, m in summary.items():
        print(f"{cond:<9}{m['n_sessions']:<9}{str(m['avg_families']):<10}{str(m['avg_families_first10']):<8}"
              f"{str(m['avg_correct']):<9}{str(m['avg_mean_novelty']):<9}{str(m['avg_correct_and_novel']):<12}"
              f"{str(m.get('avg_heuristic_score'))}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sessions": sessions, "summary": summary}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
