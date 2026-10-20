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
    ap.add_argument("--representation", default="ast", choices=["ast", "unixcoder"],
                    help="ast (main analysis) or unixcoder (the parallel representation; needs torch); "
                         "unixcoder results go to summary_unixcoder.json")
    ap.add_argument("--all-models", action="store_true",
                    help="analyse every model separately; each goes to data/results/models/<model>/summary.json "
                         "(input for scripts/model_consistency.py)")
    ap.add_argument("--model", default=None,
                    help="analyse traces from this model only (name or name@revision); default: the "
                         "model of the most recent trace")
    args = ap.parse_args()
    if args.all_models:
        from macs.analysis.aggregate import trace_models
        from macs.metrics.reference import model_slug
        names = sorted({m.split("@", 1)[0] for m in trace_models(args.traces)})
        for name in names:
            out = Path(args.out).parent / "models" / model_slug(name) / _name(args)
            print(f"\n######## {name} -> {out}")
            analyse_one(args, name, out)
        return
    out = Path(args.out)
    if args.representation != "ast" and out.name == "summary.json":
        out = out.with_name(_name(args))
    analyse_one(args, args.model, out)


def _name(args) -> str:
    return "summary.json" if args.representation == "ast" else f"summary_{args.representation}.json"


def analyse_one(args, model, out_path) -> None:

    # reference sets for the model being analysed, so novelty is computed where possible
    from macs.analysis.aggregate import resolve_model
    from macs.metrics.reference import load_references
    target = resolve_model(args.traces, model)
    rep = "embed" if args.representation == "unixcoder" else "ast"
    references = load_references(target, representation=rep) if target else {}
    if target:
        print(f"model: {target}; reference sets: {len(references)}")

    sessions, summary, pooled = analyse_dir(args.traces, backend=args.sandbox, references=references,
                                            return_pooled=True, include_legacy=args.include_old,
                                            model=model or target, representation=args.representation)
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

    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": target, "representation": args.representation,
                               "sessions": sessions, "summary": summary,
                               "pooled_diversity": pooled}, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
