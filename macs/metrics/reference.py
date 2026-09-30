"""Load a task's frozen reference set and fit the novelty yardstick.

Arush - novelty is measured RELATIVE to a set of "normal" solutions for each task. This
module turns the raw solutions that scripts/build_reference.py saved (in data/reference/)
into numbers and fits a ReferenceStats (the frozen yardstick). Two rules keep it honest:
the reference is built ONCE per task and never regenerated, and it contains only ordinary
tutor-style solutions - never the studio's own outputs - so the studio can't be scored
against itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from macs.metrics.represent import ast_feature_matrix, normalise
from macs.metrics.novelty import ReferenceStats, K_DEFAULT

REFERENCE_ROOT = Path("data/reference")


def load_reference_solutions(task_id: str, root: str | Path = REFERENCE_ROOT) -> list[str]:
    """Return the code strings saved for this task (one per reference sample)."""
    path = Path(root) / f"{task_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no reference set for {task_id}: run scripts/build_reference.py first ({path})")
    sources: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        code = rec.get("code")
        if code:
            sources.append(code)
    return sources


def _represent(sources: list[str], representation: str) -> np.ndarray:
    # Normalise first (strip comments, canonicalise names) so cosmetic differences don't
    # inflate the reference's spread. Solutions that fail to parse are dropped, not guessed.
    normed = []
    for s in sources:
        try:
            normed.append(normalise(s))
        except Exception:  # noqa: BLE001 - a model sample that doesn't parse is skipped
            continue
    if not normed:
        raise ValueError("no parseable reference solutions")
    if representation == "ast":
        return ast_feature_matrix(normed)
    if representation == "embed":
        from macs.metrics.represent import embed  # lazy - only if you ask for embeddings
        return embed(normed)
    raise ValueError(f"unknown representation {representation!r}")


def build_reference_stats(
    task_id: str,
    *,
    representation: str = "ast",
    k: int = K_DEFAULT,
    root: str | Path = REFERENCE_ROOT,
    standardise: bool | None = None,
) -> ReferenceStats:
    """Load a task's reference solutions and return a fitted, frozen novelty yardstick.

    AST features are standardised by default (each feature on its own scale); embeddings
    are already unit-norm so they are not. Pass `standardise` to override.
    """
    if standardise is None:
        standardise = representation == "ast"
    X_ref = _represent(load_reference_solutions(task_id, root), representation)
    return ReferenceStats.fit(X_ref, k=k, standardise=standardise)
