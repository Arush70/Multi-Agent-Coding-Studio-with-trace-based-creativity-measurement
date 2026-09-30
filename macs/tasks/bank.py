"""Task-bank loader.

A task is one directory under ``tasks/tier1/`` or ``tasks/tier2/`` containing
``task.yaml``::

    id: t1_two_sum
    tier: 1
    source: ASTRA            # ASTRA | MBPP | APPS | heuristic
    title: Two sum
    entry_point: two_sum
    prompt: |
      ...problem statement shown to agents...
    known_families:          # hand-written BEFORE any clustering is run (blind labels for SC3)
      - brute_force_pairs
      - sort_two_pointer
      - single_pass_hash
    public_tests:            # shown to agents
      - "assert two_sum([2, 7, 11, 15], 9) == [0, 1]"
    heldout_tests:           # NEVER shown to agents; used for scoring
      - "assert two_sum([3, 3], 6) == [0, 1]"
    scoring: tests           # tests | heuristic
    heuristic:               # only when scoring == heuristic
      baseline_score: 0.0
      instance_file: instances.json
      seeds: [1, 2, 3, 4, 5]

Held-out tests are loaded only by the scorer, never by the orchestrator, so a
bug cannot leak them into a prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

TASKS_ROOT = Path(__file__).resolve().parents[2] / "tasks"


@dataclass
class Task:
    id: str
    tier: int
    source: str
    entry_point: str
    prompt: str
    public_tests: list[str]
    title: str = ""
    source_id: str = ""          # benchmark item id (e.g. MBPP task_id, HumanEval/0)
    citation: str = ""           # full source citation for provenance (prof's requirement)
    known_families: list[str] = field(default_factory=list)
    scoring: str = "tests"       # "tests" (pass/fail) or "heuristic" (metric-scored, Tier 3)
    heuristic: dict = field(default_factory=dict)
    path: Path | None = None

    # held-out tests are deliberately not a field: see heldout_tests()

    def heldout_tests(self) -> list[str]:
        assert self.path is not None
        data = yaml.safe_load((self.path / "task.yaml").read_text())
        return list(data.get("heldout_tests", []))

    @property
    def agent_view(self) -> str:
        """Exactly what an agent is allowed to see."""
        tests = "\n".join(self.public_tests)
        return f"{self.prompt.strip()}\n\nEntry point: `{self.entry_point}`\n\nExample tests:\n{tests}"


def load_task(task_dir: str | Path) -> Task:
    p = Path(task_dir)
    data = yaml.safe_load((p / "task.yaml").read_text())
    data.pop("heldout_tests", None)
    return Task(path=p, **data)


def load_bank(root: str | Path = TASKS_ROOT, tier: int | None = None) -> list[Task]:
    root = Path(root)
    dirs = sorted(d for d in root.glob("tier*/*") if (d / "task.yaml").exists())
    tasks = [load_task(d) for d in dirs]
    if tier is not None:
        tasks = [t for t in tasks if t.tier == tier]
    return tasks
