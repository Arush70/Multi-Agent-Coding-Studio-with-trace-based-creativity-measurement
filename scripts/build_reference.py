"""Generate the frozen reference set of ordinary solutions for each task.

The model is asked for many ordinary solutions to each task and they are saved to
data/reference/<task_id>.jsonl. Novelty is measured as distance from this set, so it is built
once, with the same model as the experiment, and never regenerated.

    python scripts/build_reference.py --tasks all --n 100 --profile local --sandbox subprocess
    python scripts/build_reference.py --tasks tier3            # one tier
    python scripts/build_reference.py --tasks t1_mbpp19_test_duplicate,t3_bin_packing

Notes:
  * Existing files are topped up, never rewritten: a task with 30 samples gets seeds 30..99
    appended. Because seeds are fixed, topping 30 up to 100 gives exactly the same set as
    building 100 from scratch. A task already at --n is skipped. --overwrite rebuilds.
  * Safe to stop and restart: each sample is written as soon as it is made.
  * Sensitivity check (proposal: 50 and 200): build --n 200 on selected tasks; the analysis
    still uses only the first 100 (macs.metrics.reference.REFERENCE_N), and
    scripts/reference_sensitivity.py compares 50 / 100 / 200.
  * Seeds are fixed (REF_SEED_BASE + i) so the whole set is reproducible.
  * The prompt is a neutral "write a correct solution", not a studio role, because the
    reference is meant to contain typical baseline solutions.
  * On a small GPU, a quick check with --n 20 is sensible before the full 100-per-task run.
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


def _is_dir(spec: str) -> bool:
    try:
        return Path(spec).is_dir()
    except OSError:                   # e.g. a long list of ids is not a valid path
        return False


def resolve_tasks(spec: str) -> list:
    """'all' | 'tier1' | 'tier2' | 'tier3' | 'main' | 'contamination' | 'open' | a task directory |
    comma-separated task ids. 'all' and the tiers cover the experiment's tasks (the 24-task main
    bank and the contamination check); open problems for the search study are chosen with 'open'
    or by id."""
    if spec in ("all", "tier1", "tier2", "tier3"):
        return [t for t in load_bank(tier=None if spec == "all" else int(spec[-1])) if t.task_set != "open_problem"]
    if spec in ("main", "contamination", "open"):
        want = {"main": "main", "contamination": "contamination_check", "open": "open_problem"}[spec]
        return [t for t in load_bank() if t.task_set == want]
    if "," not in spec and _is_dir(spec):
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


def _other_model_rows(path: Path, model: str, revision: str) -> int:
    """Count samples in `path` that were not made by model@revision (rows without a model count)."""
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model") != model or row.get("revision", "") != revision:
            n += 1
    return n


def build_task_reference(task, client, *, n: int, path: Path, sandbox: str = "subprocess",
                         temperature: float = 0.7, overwrite: bool = False, log=print,
                         parallel: int = 1) -> int:
    """Make sure `path` holds reference samples for seeds REF_SEED_BASE + 0..n-1. Returns how
    many new samples were written. Never rewrites an existing sample unless overwrite=True."""
    from macs.metrics.represent import ast_features  # to record how many parse

    if overwrite and path.exists():
        path.write_text("", encoding="utf-8")
    mismatched = _other_model_rows(path, client.profile.model, client.profile.revision)
    if mismatched:
        raise RuntimeError(
            f"{path} holds {mismatched} sample(s) from a different model than "
            f"{client.profile.model}@{client.profile.revision}. A reference set must come from one "
            f"model; move the file aside (e.g. to data/reference_archive/) or pass --overwrite.")
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

    def make(i: int) -> tuple[dict, bool]:
        """One reference sample: a model call, then the public tests. Independent of the others,
        so several can run at once (the seed fixes each sample, whatever the order)."""
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
            parsed = True
        except Exception:  # noqa: BLE001
            parsed = False
        return {
            "task_id": task.id, "seed": seed, "code": src,
            "code_hash": res.code_hash,
            "public_passed": res.tests_passed, "public_failed": res.tests_failed,
            "temperature": temperature,
            "model": client.profile.model, "revision": client.profile.revision,
        }, parsed

    n_parse = 0
    with path.open("a", encoding="utf-8") as fh:
        def write(row: dict, parsed: bool, j: int) -> None:
            nonlocal n_parse
            n_parse += parsed
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()                            # written as soon as made: safe to stop
            if j % 10 == 0:
                log(f"  {task.id}: {len(have) + j}/{n}")

        if parallel <= 1:
            for j, i in enumerate(todo, start=1):
                write(*make(i), j)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [pool.submit(make, i) for i in todo]
                for j, fut in enumerate(as_completed(futures), start=1):
                    write(*fut.result(), j)
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
    ap.add_argument("--parallel", type=int, default=1,
                    help="samples made at the same time (a server with several slots, e.g. on ISCA)")
    args = ap.parse_args()

    from macs.metrics.reference import reference_path
    client = LLMClient.from_config(profile=args.profile)
    for task in resolve_tasks(args.tasks):
        # each model has its own reference set (see macs.metrics.reference.reference_path)
        path = reference_path(task.id, client.profile.model, root=args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        build_task_reference(task, client, n=args.n, path=path,
                             sandbox=args.sandbox, temperature=args.temperature, overwrite=args.overwrite,
                             parallel=args.parallel)


if __name__ == "__main__":
    main()
