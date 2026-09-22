"""Run one session against your local model and write a trace. YOUR FIRST REAL DATA.

Arush - usage (Ollama must be running; you are on condition A or C for now):

    python scripts/run_session.py --task tasks/tier1/t1_two_sum --condition A --seed 1 --sandbox subprocess

Use --sandbox subprocess for now (Docker isn't up yet). Switch to --sandbox docker for
the real experiment once Docker works - that is the only backend whose numbers you report.
The script writes data/traces/<id>.jsonl, validates it, and prints it so you can see the
learner turn, the tutor's solution, and the sandbox verdict.
"""
import argparse
from pathlib import Path

from macs.llm.client import LLMClient
from macs.tasks.bank import load_task
from macs.orchestrator import SessionConfig, run_session
from macs.trace.writer import validate_file, read_trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="path to a task dir, e.g. tasks/tier1/t1_two_sum")
    ap.add_argument("--condition", default="A", choices=["A", "C"], help="A=1 learner, C=2 learners (studio B/D/D_minus come later)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--profile", default="local", help="model profile in config/models.yaml")
    ap.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"],
                    help="subprocess = dev (works now); docker = the real experiment backend")
    ap.add_argument("--out", default="data/traces")
    args = ap.parse_args()

    if args.sandbox == "subprocess":
        print("[note] sandbox=subprocess is DEVELOPMENT ONLY (no memory isolation). "
              "Use --sandbox docker for results you report.\n")

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
