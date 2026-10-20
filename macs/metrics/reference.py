"""Load a task's frozen reference set and fit the novelty yardstick.

Novelty is measured relative to a set of ordinary solutions for each task. This module turns
the solutions saved by scripts/build_reference.py (in data/reference/) into feature vectors
and fits a ReferenceStats. The reference is built once per task and never regenerated, and it
contains only ordinary solutions, never the studio's own outputs, so the studio is never
scored against itself. The analysis uses the first REFERENCE_N samples by seed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from macs.metrics.represent import ast_feature_matrix, normalise
from macs.metrics.novelty import ReferenceStats, K_DEFAULT

REFERENCE_ROOT = Path("data/reference")
# Proposal: 100 reference solutions per task, with sensitivity checked at 50 and 200. The analysis
# always uses the FIRST REFERENCE_N samples (by seed order), so building 200 for the sensitivity
# check never changes the main result. Fewer than this -> still usable, but flagged.
REFERENCE_N = 100


def model_slug(model: str) -> str:
    """A folder name for a model id, e.g. 'qwen3.5:9b' -> 'qwen3.5_9b'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def _first_model(path: Path) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return json.loads(line).get("model")
    except (OSError, json.JSONDecodeError):
        return None
    return None


def reference_path(task_id: str, model: str | None = None, root: str | Path | None = None) -> Path:
    """Where a task's reference set for `model` lives.

    Each model has its own reference set, because novelty is measured against what THAT model
    ordinarily writes. The first model's sets live in data/reference/<task>.jsonl (where they were
    built before several models were supported); every other model's sets live in
    data/reference/models/<model>/<task>.jsonl. With model=None the first location is used.
    """
    root = Path(REFERENCE_ROOT if root is None else root)
    legacy = root / f"{task_id}.jsonl"
    if model is None:
        return legacy
    if legacy.exists() and _first_model(legacy) == model:
        return legacy
    return root / "models" / model_slug(model) / f"{task_id}.jsonl"


def reference_tasks(model: str | None = None, root: str | Path | None = None) -> list[str]:
    """Task ids that have a reference file for `model` (None: the first location)."""
    root = Path(REFERENCE_ROOT if root is None else root)
    found = set()
    for f in root.glob("*.jsonl"):
        if model is None or _first_model(f) == model:
            found.add(f.stem)
    if model is not None:
        found |= {f.stem for f in (root / "models" / model_slug(model)).glob("*.jsonl")}
    return sorted(found)


def load_references(model: str | None = None, *, representation: str = "ast", log=print) -> dict:
    """ReferenceStats for every task that has a reference set for `model`, keyed by task id.
    Sets with fewer than REFERENCE_N parseable samples are used but reported."""
    refs, small = {}, []
    for tid in reference_tasks(model):
        try:
            refs[tid] = build_reference_stats(tid, model=model, representation=representation)
        except Exception:  # noqa: BLE001 - an empty or unparseable file gives no novelty for that task
            continue
        if len(refs[tid].X_ref) < REFERENCE_N:
            small.append(f"{tid} ({len(refs[tid].X_ref)})")
    if small and log:
        log(f"[warn] reference sets below {REFERENCE_N} parseable samples (novelty less reliable): "
            + ", ".join(small))
    return refs


def load_reference_solutions(task_id: str, root: str | Path | None = None, n: int | None = REFERENCE_N,
                             model: str | None = None) -> list[str]:
    """Return the code strings saved for this task (one per reference sample), the first `n`
    by seed order (n=None -> all of them)."""
    path = reference_path(task_id, model, root)
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
    root: str | Path | None = None,
    standardise: bool | None = None,
    n: int | None = REFERENCE_N,
    model: str | None = None,
) -> ReferenceStats:
    """Load a task's reference solutions and return a fitted, frozen novelty yardstick.

    AST features are standardised by default (each feature on its own scale); embeddings
    are already unit-norm so they are not. Pass `standardise` to override.
    """
    if standardise is None:
        standardise = representation == "ast"
    X_ref = _represent(load_reference_solutions(task_id, root, n=n, model=model), representation)
    return ReferenceStats.fit(X_ref, k=k, standardise=standardise)
