"""Run one session against the configured model and write a trace.

    python scripts/run_session.py --task tasks/tier1/t1_mbpp19_test_duplicate --condition A --seed 1 --sandbox subprocess

The subprocess sandbox is for development; reported results use --sandbox docker. The
script writes data/traces/<id>.jsonl, validates it, and prints the events. For many
sessions at once use scripts/run_experiment.py.
"""
import argparse
from pathlib import Path

from macs.llm.client import LLMClient
from macs.tasks.bank import load_task
from macs.orchestrator import SessionConfig, check_facilitator_dependencies, run_session
from macs.trace.writer import validate_file, read_trace
from macs import conditions as conds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="path to a task dir, e.g. tasks/tier1/t1_two_sum")
    ap.add_argument("--condition", default="A", choices=conds.ALL,
                    help="; ".join(f"{c} = {v.description}" for c, v in conds.CONDITIONS.items()))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--profile", default="local", help="model profile in config/models.yaml")
    ap.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"],
                    help="subprocess = dev (works now); docker = the real experiment backend")
    ap.add_argument("--out", default="data/traces")
    args = ap.parse_args()

    if args.sandbox == "subprocess":
        print("[note] sandbox=subprocess is DEVELOPMENT ONLY (no memory isolation). "
              "Use --sandbox docker for results you report.\n")

    if conds.get(args.condition).is_studio:
        problem = check_facilitator_dependencies()
        if problem:
            raise SystemExit(problem)

    client = LLMClient.from_config(profile=args.profile)
    task = load_task(args.task)
    cfg = SessionConfig(condition=args.condition, seed=args.seed, sandbox_backend=args.sandbox)

    print(f"running {task.id}  condition={args.condition}  seed={args.seed}  model={client.profile.model}")
    path = run_session(task, cfg, client, out_dir=args.out)

    errors = validate_file(path)
    print(f"\ntrace written: {path}")
    print(f"validation: {'OK, 0 errors' if not errors else errors}\n")
    for ev in read_trace(path):
        line = f"seq {ev['seq']:>2}  {ev['role']:<10} {ev['event']:<14}"
        if ev["event"] == "verification":
            line += f"  passed {ev['tests_passed']}/{ev['tests_passed']+ev['tests_failed']}"
        elif ev.get("content"):
            line += "  " + ev["content"].replace("\n", " ")[:60]
        print(line)


if __name__ == "__main__":
    main()
