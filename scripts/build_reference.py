"""Generate the frozen reference set of 'normal' solutions for each task.

Arush - this asks the model for many ordinary solutions to each task and saves them to
data/reference/<task_id>.jsonl. Novelty is later measured as distance FROM this set, so it
is built once, on the same model you run the experiment with, and never regenerated.

    python scripts/build_reference.py --tasks all --n 100 --profile local --sandbox subprocess
    python scripts/build_reference.py --tasks tier3            # one tier
    python scripts/build_reference.py --tasks t1_mbpp19_test_duplicate,t3_bin_packing

Notes:
  * Existing files are TOPPED UP, never rewritten: a task with 30 samples gets seeds 30..99
    appended. Because seeds are fixed, topping 30 up to 100 gives exactly the same set as
    building 100 from scratch. A task already at --n is skipped. --overwrite rebuilds.
  * Safe to stop and restart: each sample is written as soon as it is made.
  * Sensitivity check (proposal: 50 and 200): build --n 200 on the tasks you want; the analysis
    still uses only the first 100 (macs.metrics.reference.REFERENCE_N), and
    scripts/reference_sensitivity.py compares 50 / 100 / 200.
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


def resolve_tasks(spec: str) -> list:
    """'all' | 'tier1' | 'tier2' | 'tier3' | a task directory | comma-separated task ids."""
    if spec in ("all", "tier1", "tier2", "tier3"):
        return load_bank(tier=None if spec == "all" else int(spec[-1]))
    if Path(spec).is_dir():
        return [load_task(spec)]
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    bank = {t.id: t for t in load_bank()}
    missing = [w for w in wanted if w not in bank]
    if missing:
        raise SystemExit(f"unknown task id(s): {', '.join(missing)}")
    return [bank[w] for w in wanted]


def existing_seeds(path: Path) -> list[int]:
    if not path.exists():
        return []
    seeds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            seeds.append(int(json.loads(line)["seed"]))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue  # a half-written last line from an interrupted run: ignored, re-made below
    return seeds


def build_task_reference(task, client, *, n: int, path: Path, sandbox: str = "subprocess",
                         temperature: float = 0.7, overwrite: bool = False, log=print) -> int:
    """Make sure `path` holds reference samples for seeds REF_SEED_BASE + 0..n-1. Returns how
    many new samples were written. Never rewrites an existing sample unless overwrite=True."""
    from macs.metrics.represent import ast_features  # to record how many parse

    if overwrite and path.exists():
        path.write_text("", encoding="utf-8")
    have = set(existing_seeds(path))
    todo = [i for i in range(n) if REF_SEED_BASE + i not in have]
    if not todo:
        log(f"skip {task.id} (already has {len(have)} >= {n} samples)")
        return 0
    # drop a trailing partial line (interrupted run) so appended lines stay valid JSONL
    if path.exists():
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        good = []
        for l in lines:
            try:
                json.loads(l); good.append(l)
            except json.JSONDecodeError:
                pass
        if len(good) != len(lines):
            path.write_text("".join(g + "\n" for g in good), encoding="utf-8")

    n_parse = 0
    with path.open("a", encoding="utf-8") as fh:
        for j, i in enumerate(todo, start=1):
            seed = REF_SEED_BASE + i
            messages = [
                {"role": "system", "content": REFERENCE_PROMPT},
                {"role": "user", "content": task.agent_view},
            ]
            comp = client.chat(messages, temperature=temperature, seed=seed)
            src = extract_code(comp.text)
            # run on PUBLIC tests only (never held-out) so we can note which ones work
            res = runner.run(src, task.public_tests, backend=sandbox, cpu_seconds=5)
            try:
                ast_features(src)
                n_parse += 1
            except Exception:  # noqa: BLE001
                pass
            fh.write(json.dumps({
                "task_id": task.id, "seed": seed, "code": src,
                "code_hash": res.code_hash,
                "public_passed": res.tests_passed, "public_failed": res.tests_failed,
                "temperature": temperature,
                "model": client.profile.model, "revision": client.profile.revision,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            if j % 10 == 0:
                log(f"  {task.id}: {len(have) + j}/{n}")
    log(f"built {task.id}: +{len(todo)} samples ({n_parse} parse) -> {path} (now {len(have) + len(todo)})")
    return len(todo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all",
                    help="'all', 'tier1', 'tier2', 'tier3', a task dir path, or comma-separated task ids")
    ap.add_argument("--n", type=int, default=100, help="reference samples per task (proposal: 100)")
    ap.add_argument("--profile", default="local")
    ap.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"])
    ap.add_argument("--out", default="data/reference")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--overwrite", action="store_true", help="rebuild from scratch even if samples exist")
    args = ap.parse_args()

    client = LLMClient.from_config(profile=args.profile)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in resolve_tasks(args.tasks):
        build_task_reference(task, client, n=args.n, path=out_dir / f"{task.id}.jsonl",
                             sandbox=args.sandbox, temperature=args.temperature, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
