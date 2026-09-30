"""Generate the frozen reference set of 'normal' solutions for each task.

Arush - this asks the model for many ordinary solutions to each task and saves them to
data/reference/<task_id>.jsonl. Novelty is later measured as distance FROM this set, so it
is built once, on the same model you run the experiment with, and never regenerated.

    python scripts/build_reference.py --tasks all --n 100 --profile local --sandbox subprocess

Notes:
  * A task whose file already exists is SKIPPED (the set is frozen). Use --overwrite to force.
  * Seeds are fixed (REF_SEED_BASE + i) so the whole set is reproducible.
  * The prompt is a neutral "write a correct solution" - NOT a studio agent - because the
    reference is meant to be typical, baseline solutions.
  * On your 8 GB laptop start small (e.g. --n 20) to check the loop; the full 100-per-task
    run belongs on the experiment GPU.
"""
import argparse
import json
from pathlib import Path

from macs.llm.client import LLMClient
from macs.tasks.bank import load_task, load_bank
from macs.orchestrator import extract_code
from macs.sandbox import runner

REF_SEED_BASE = 100_000

REFERENCE_PROMPT = (
    "You are an experienced Python programmer. Write one correct, self-contained solution to the "
    "task below. Return only a single ```python code block that defines the required function. "
    "Do not explain."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all", help="'all', 'tier1', 'tier2', or a task dir path")
    ap.add_argument("--n", type=int, default=100, help="reference samples per task")
    ap.add_argument("--profile", default="local")
    ap.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"])
    ap.add_argument("--out", default="data/reference")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--overwrite", action="store_true", help="rebuild even if a set already exists")
    args = ap.parse_args()

    client = LLMClient.from_config(profile=args.profile)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # resolve which tasks to build
    if args.tasks in ("all", "tier1", "tier2"):
        tier = 1 if args.tasks == "tier1" else 2 if args.tasks == "tier2" else None
        tasks = load_bank(tier=tier)
    else:
        tasks = [load_task(args.tasks)]

    from macs.metrics.represent import ast_features  # to record how many parse

    for task in tasks:
        path = out_dir / f"{task.id}.jsonl"
        if path.exists() and not args.overwrite:
            print(f"skip {task.id} (already frozen: {path})")
            continue

        n_parse = 0
        with path.open("w", encoding="utf-8") as fh:
            for i in range(args.n):
                seed = REF_SEED_BASE + i
                messages = [
                    {"role": "system", "content": REFERENCE_PROMPT},
                    {"role": "user", "content": task.agent_view},
                ]
                comp = client.chat(messages, temperature=args.temperature, seed=seed)
                src = extract_code(comp.text)
                # run on PUBLIC tests only (never held-out) so we can note which ones work
                res = runner.run(src, task.public_tests, backend=args.sandbox, cpu_seconds=5)
                try:
                    ast_features(src)
                    n_parse += 1
                except Exception:  # noqa: BLE001
                    pass
                fh.write(json.dumps({
                    "task_id": task.id, "seed": seed, "code": src,
                    "code_hash": res.code_hash,
                    "public_passed": res.tests_passed, "public_failed": res.tests_failed,
                    "temperature": args.temperature,
                    "model": client.profile.model, "revision": client.profile.revision,
                }, ensure_ascii=False) + "\n")
                if (i + 1) % 10 == 0:
                    print(f"  {task.id}: {i + 1}/{args.n}")
        print(f"built {task.id}: {args.n} samples ({n_parse} parse) -> {path}")


if __name__ == "__main__":
    main()
