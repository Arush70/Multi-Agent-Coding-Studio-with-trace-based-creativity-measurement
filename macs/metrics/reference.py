"""Load a task's frozen reference set and fit the novelty yardstick.

Novelty is measured relative to a set of ordinary solutions for each task. This module turns
the solutions saved by scripts/build_reference.py (in data/reference/) into feature vectors
and fits a ReferenceStats. The reference is built once per task and never regenerated, and it
contains only ordinary solutions, never the studio's own outputs, so the studio is never
scored against itself. The analysis uses the first REFERENCE_N samples by seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from macs.metrics.represent import ast_feature_matrix, normalise
from macs.metrics.novelty import ReferenceStats, K_DEFAULT

REFERENCE_ROOT = Path("data/reference")
# Proposal: 100 reference solutions per task, with sensitivity checked at 50 and 200. The analysis
# always uses the FIRST REFERENCE_N samples (by seed order), so building 200 for the sensitivity
# check never changes the main result. Fewer than this -> still usable, but flagged.
REFERENCE_N = 100


def load_reference_solutions(task_id: str, root: str | Path = REFERENCE_ROOT, n: int | None = REFERENCE_N) -> list[str]:
    """Return the code strings saved for this task (one per reference sample), the first `n`
    by seed order (n=None -> all of them)."""
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
            sources.append((rec.get("seed", 0), code))
    sources.sort(key=lambda t: t[0])          # seed order, so 'first n' is reproducible
    codes = [c for _, c in sources]
    return codes if n is None else codes[:n]


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
        from macs.metrics.represent import embed  # lazy - loaded only when embeddings are requested
        return embed(normed)
    raise ValueError(f"unknown representation {representation!r}")


def build_reference_stats(
    task_id: str,
    *,
    representation: str = "ast",
    k: int = K_DEFAULT,
    root: str | Path = REFERENCE_ROOT,
    standardise: bool | None = None,
    n: int | None = REFERENCE_N,
) -> ReferenceStats:
    """Load a task's reference solutions and return a fitted, frozen novelty yardstick.

    AST features are standardised by default (each feature on its own scale); embeddings
    are already unit-norm so they are not. Pass `standardise` to override.
    """
    if standardise is None:
        standardise = representation == "ast"
    X_ref = _represent(load_reference_solutions(task_id, root, n=n), representation)
    return ReferenceStats.fit(X_ref, k=k, standardise=standardise)
