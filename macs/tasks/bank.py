"""Task-bank loader.

Arush - this file exists to enforce ONE rule that, if broken, invalidates your whole
experiment: an agent must never see a held-out test. Your novelty and usefulness
numbers only mean something if the model solved the task without having seen the
answers it is judged on. So the design here is deliberately defensive:

  * When a task is loaded, the held-out tests are STRIPPED OUT and never put on the
    Task object. There is no attribute an agent-facing code path could accidentally
    read them from.
  * The only way to get them is to call task.heldout_tests(), which re-reads the yaml
    from disk. The scorer calls this; the orchestrator never does. If you ever see
    heldout_tests() called anywhere near prompt-building code, that is a bug to stop
    and fix.
  * agent_view is the ONLY thing an agent is shown: prompt + entry point + PUBLIC
    tests. Nothing else.

A task is one directory under tasks/tier1/ or tasks/tier2/ containing task.yaml.
Its fields are documented in the two example tasks shipped alongside this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# tasks/ sits at the repo root, two levels up from this file (macs/tasks/bank.py).
TASKS_ROOT = Path(__file__).resolve().parents[2] / "tasks"


@dataclass
class Task:
    id: str
    tier: int
    source: str
    title: str
    entry_point: str
    prompt: str
    public_tests: list[str]
    known_families: list[str] = field(default_factory=list)
    scoring: str = "tests"
    heuristic: dict = field(default_factory=dict)
    path: Path | None = None

    # NOTE: there is intentionally no `heldout_tests` field here. Making it a method
    # that re-reads the file (below) means the answers never sit in memory on an object
    # that prompt-building code touches. This is a guardrail, not an accident.

    def heldout_tests(self) -> list[str]:
        # Re-reads task.yaml from disk on demand. ONLY the scorer should call this.
        assert self.path is not None, "task has no path; load it via load_task/load_bank"
        data = yaml.safe_load((self.path / "task.yaml").read_text())
        return list(data.get("heldout_tests", []))

    @property
    def agent_view(self) -> str:
        """Exactly what an agent is allowed to see - and nothing more. If you ever need
        to change what agents see, change it HERE, in one place, so the boundary stays
        auditable."""
        tests = "\n".join(self.public_tests)
        return f"{self.prompt.strip()}\n\nEntry point: `{self.entry_point}`\n\nExample tests:\n{tests}"


def load_task(task_dir: str | Path) -> Task:
    p = Path(task_dir)
    data = yaml.safe_load((p / "task.yaml").read_text())
    data.pop("heldout_tests", None)   # <-- the strip. Held-out tests never reach the Task object.
    return Task(path=p, **data)


def load_bank(root: str | Path = TASKS_ROOT, tier: int | None = None) -> list[Task]:
    """Load every task directory, optionally filtered to one tier. Sorted by path so
    the order is stable across machines (matters for reproducible seeds)."""
    root = Path(root)
    dirs = sorted(d for d in root.glob("tier*/*") if (d / "task.yaml").exists())
    tasks = [load_task(d) for d in dirs]
    if tier is not None:
        tasks = [t for t in tasks if t.tier == tier]
    return tasks
