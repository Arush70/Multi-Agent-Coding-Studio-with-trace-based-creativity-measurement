"""Run many sessions in one go: tasks x conditions x seeds. Safe to stop and restart.

The model server must be running.

    # pilot: a few tasks, all five conditions, 3 seeds
    python scripts/run_experiment.py --tasks t1_mbpp19_test_duplicate,t2_humaneval0_has_close_elements,t3_bin_packing --seeds 1-3

    # main run (proposal: 6 repetitions per cell), reported numbers on docker
    python scripts/run_experiment.py --tasks all --seeds 1-6 --sandbox docker

    # see what would run without running it
    python scripts/run_experiment.py --tasks all --seeds 1-6 --dry-run

A (task, condition, seed) that already has a complete trace from the current orchestrator
version is skipped, so re-running the same command continues where it stopped. A failing
session is reported and the batch carries on. Afterwards: python scripts/analyze.py
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_reference import resolve_tasks  # noqa: E402  (same task selection syntax)
from macs.llm.client import LLMClient  # noqa: E402
from macs.metrics.reference import REFERENCE_ROOT, REFERENCE_N  # noqa: E402
from macs.orchestrator import (  # noqa: E402
    ORCHESTRATOR_VERSION, SessionConfig, check_facilitator_dependencies, is_comparable, run_session)
from macs import conditions as conds  # noqa: E402
from macs.trace.writer import validate_file, read_trace  # noqa: E402

CONDITIONS = conds.ALL            # every known condition; the default run uses conds.CORE
PAUSE_AFTER, PAUSE_SECONDS, STOP_AFTER = 3, 60, 6


def parse_seeds(spec: str) -> list[int]:
    """'1-6' -> [1..6]; '1,3,5' -> [1,3,5]; '2' -> [2]."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def done_cells(traces_dir: Path, model: str | None = None) -> set[tuple[str, str, int]]:
    """(task, condition, seed) cells that already have a complete trace from this orchestrator
    (and, when `model` is given, from that model: a cell done with another model is not done)."""
    done = set()
    for p in traces_dir.glob("*.jsonl"):
        try:
            events = list(read_trace(p))
        except (OSError, json.JSONDecodeError):
            continue
        if not events or events[0].get("event") != "session_start" or events[-1].get("event") != "session_end":
            continue
        # a session that crashed (e.g. Ollama went down) still gets a session_end, but without
        # 'completed' / 'stopped' - it is not done and will be run again
        if not str(events[-1].get("content", "")).startswith(("completed", "stopped")):
            continue
        h = events[0].get("header", {})
        if not is_comparable(h.get("orchestrator_version")):
            continue
        if model is not None and h.get("model") != model:
            continue
        done.add((events[0]["task_id"], events[0]["condition"], int(h.get("seed", -1))))
    return done


def profile_model(profile: str, config: str = "config/models.yaml") -> str | None:
    """The model name a profile in config/models.yaml points at (None if it cannot be read)."""
    try:
        import yaml
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        return cfg["profiles"][profile]["model"]
    except Exception:  # noqa: BLE001 - no config yet: fall back to version-only matching
        return None


def reference_status(task_id: str, model: str | None = None) -> str:
    from macs.metrics.reference import reference_path
    p = reference_path(task_id, model)
    if not p.exists():
        return "no reference (novelty off, default tau)"
    n = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    return f"reference {n}" + ("" if n >= REFERENCE_N else f" < {REFERENCE_N}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all", help="'all', 'tier1/2/3', a task dir, or comma-separated ids")
    ap.add_argument("--conditions", default=",".join(conds.CORE),
                    help="comma-separated; 'all' for every condition (" + ",".join(CONDITIONS) + ")")
    ap.add_argument("--seeds", default="1-6", help="e.g. 1-6 or 1,2,3 (proposal: 6 repetitions)")
    ap.add_argument("--profile", default="local",
                    help="model profile in config/models.yaml; several comma-separated profiles run one "
                         "after another (each model is analysed separately, never pooled)")
    ap.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"])
    ap.add_argument("--out", default="data/traces")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = resolve_tasks(args.tasks)
    conditions = CONDITIONS if args.conditions.strip() == "all" else \
        [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown condition(s): {', '.join(bad)}")
    seeds = parse_seeds(args.seeds)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    profiles = [p.strip() for p in args.profile.split(",") if p.strip()]
    for profile in profiles:
        # failed sessions are listed and retried on the next run; they do not stop the other models
        run_profile(profile, tasks, conditions, seeds, out, args.sandbox, args.dry_run)


def run_profile(profile, tasks, conditions, seeds, out, sandbox, dry_run) -> list:
    model = profile_model(profile)
    done = done_cells(out, model=model)
    plan = [(t, c, s) for t in tasks for s in seeds for c in conditions if (t.id, c, s) not in done]
    total = len(tasks) * len(conditions) * len(seeds)
    print(f"== profile {profile} (model {model}) ==")
    print(f"{len(tasks)} task(s) x {len(conditions)} condition(s) x {len(seeds)} seed(s) = {total} sessions; "
          f"{total - len(plan)} already done, {len(plan)} to run  (orchestrator {ORCHESTRATOR_VERSION})")
    for t in tasks:
        print(f"  {t.id:<40} {reference_status(t.id, model)}")
    if sandbox == "subprocess":
        print("[note] sandbox=subprocess is for development; report numbers from --sandbox docker.")
    if dry_run or not plan:
        return []
    if any(conds.get(c).is_studio for _, c, _ in plan):
        problem = check_facilitator_dependencies()
        if problem:
            raise SystemExit(problem)

    client = LLMClient.from_config(profile=profile)
    failures = []
    consecutive = 0
    t0 = time.time()
    for i, (task, cond, seed) in enumerate(plan, start=1):
        label = f"[{i}/{len(plan)}] {task.id} {cond} seed={seed}"
        try:
            path = run_session(task, SessionConfig(condition=cond, seed=seed, sandbox_backend=sandbox),
                               client, out_dir=out)
            errors = validate_file(path)
            end = list(read_trace(path))[-1].get("content", "")
            print(f"{label}  ok  {end}" + (f"  VALIDATION: {errors[:2]}" if errors else ""))
            consecutive = 0
        except KeyboardInterrupt:
            print("\nstopped by user; re-run the same command to continue")
            break
        except Exception as e:  # noqa: BLE001 - one bad session must not kill the batch
            failures.append((label, repr(e)))
            print(f"{label}  FAILED  {e!r}")
            traceback.print_exc(limit=2)
            consecutive += 1
            # several failures in a row usually mean the model server is down: wait, then give up
            # rather than marking every remaining session as failed
            if consecutive == PAUSE_AFTER:
                print(f"      {consecutive} failures in a row; waiting {PAUSE_SECONDS}s for the model server ...")
                time.sleep(PAUSE_SECONDS)
            elif consecutive >= STOP_AFTER:
                print(f"\n{consecutive} failures in a row: stopping this model's run. Check the model server "
                      "(Setup tab), then start the same run again; it continues where it stopped.")
                break
        elapsed = time.time() - t0
        print(f"      elapsed {elapsed/60:.1f} min, about {elapsed / i * (len(plan) - i) / 60:.1f} min left")
    if failures:
        print(f"\n{len(failures)} session(s) failed (they will be retried on the next run):")
        for label, err in failures:
            print(f"  {label}: {err}")
    print("\nnext: python scripts/analyze.py --sandbox", sandbox, "--model", model or "<model>")
    return failures


if __name__ == "__main__":
    main()
