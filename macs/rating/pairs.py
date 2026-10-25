"""Build the paired-comparison set for the rating study (SC4).

Each pair is two CORRECT solutions (they pass every held-out test) to the same task, so raters
judge originality and creativity, never correctness. Pairs are drawn to span a range of
automatic novelty differences, so the human ranking can be compared with the automatic score.
Everything is seeded and written to data/rating/pairs.json; what raters see contains only the
task and the two codes (comments and docstrings removed), never the condition or the score.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

RATING_DIR = Path("data/rating")
N_PAIRS = 60
MAX_USES = 2                 # a solution appears in at most this many pairs


@dataclass
class Solution:
    sid: str                 # stable id: task|session|proposal
    task_id: str
    session: str
    proposal_id: str
    condition: str
    code: str                # as written by the model
    display: str             # what raters see (blind view)
    novelty: float           # automatic novelty z (AST space)


def collect_solutions(traces_dir="data/traces", *, model: str | None = None, task_set: str = "main",
                      backend: str = "subprocess", log=print) -> list[Solution]:
    """Every distinct correct solution in the comparable traces of one model, with its novelty."""
    from macs.analysis.aggregate import resolve_model, _embed
    from macs.analysis.score import implementations, score_code_heldout, task_by_id
    from macs.metrics.reference import load_references
    from macs.metrics.represent import normalise
    from macs.orchestrator import is_comparable
    from macs.trace.writer import read_trace
    from macs.ui.pipeline import blind_view

    model = resolve_model(traces_dir, model)
    refs = load_references(model, log=None)
    seen, out = set(), []
    for p in sorted(Path(traces_dir).glob("*.jsonl")):
        events = list(read_trace(p))
        start = events[0] if events else {}
        h = start.get("header", {})
        if start.get("event") != "session_start" or h.get("model") != model:
            continue
        if not is_comparable(h.get("orchestrator_version")):
            continue
        try:
            task = task_by_id(start["task_id"])
        except KeyError:
            continue
        if task.scoring != "tests" or (task_set != "all" and task.task_set != task_set):
            continue
        ref = refs.get(task.id)
        if ref is None:
            continue
        for pid, code in implementations(events):
            try:
                key = (task.id, normalise(code))
            except Exception:  # noqa: BLE001 - unparseable code cannot be rated
                continue
            if key in seen or not score_code_heldout(task, code, backend=backend).get("passed_all"):
                continue
            X, keep = _embed([code])
            if not keep:
                continue
            z = float(ref.novelty(X)[0])
            if z != z:                                    # NaN: the task's reference has no spread
                continue
            seen.add(key)
            out.append(Solution(sid=f"{task.id}|{p.stem}|{pid}", task_id=task.id, session=p.stem, proposal_id=pid,
                                condition=start["condition"], code=code, display=blind_view(code),
                                novelty=z))
    if log:
        log(f"{len(out)} distinct correct solutions from {len({s.task_id for s in out})} task(s), model {model}")
    return out


def build_pairs(solutions: list[Solution], n_pairs: int = N_PAIRS, seed: int = 2026) -> list[dict]:
    """Choose n_pairs pairs: spread evenly over tasks, spanning small to large novelty differences,
    each solution used at most MAX_USES times, A/B order randomised."""
    rng = random.Random(seed)
    by_task: dict[str, list[Solution]] = {}
    for s in solutions:
        by_task.setdefault(s.task_id, []).append(s)
    by_task = {t: v for t, v in by_task.items() if len(v) >= 2}
    if not by_task:
        return []
    # candidate pairs per task, ordered so that novelty gaps from small to large are all covered
    cands = {}
    for t, sols in by_task.items():
        pairs = list(itertools.combinations(sorted(sols, key=lambda s: s.sid), 2))
        rng.shuffle(pairs)
        gaps = np.array([abs(a.novelty - b.novelty) for a, b in pairs])
        order = np.argsort(gaps, kind="stable")
        # interleave: largest, smallest, second largest, second smallest, ...
        idx, lo, hi = [], 0, len(order) - 1
        while lo <= hi:
            idx.append(order[hi]); hi -= 1
            if lo <= hi:
                idx.append(order[lo]); lo += 1
        cands[t] = [pairs[i] for i in idx]
    uses: dict[str, int] = {}
    chosen = []
    tasks = sorted(cands)
    while len(chosen) < n_pairs and any(cands.values()):
        for t in tasks:
            while cands[t]:
                a, b = cands[t].pop(0)
                if uses.get(a.sid, 0) < MAX_USES and uses.get(b.sid, 0) < MAX_USES:
                    uses[a.sid] = uses.get(a.sid, 0) + 1
                    uses[b.sid] = uses.get(b.sid, 0) + 1
                    chosen.append((a, b))
                    break
            if len(chosen) >= n_pairs:
                break
    out = []
    for i, (a, b) in enumerate(chosen, start=1):
        if rng.random() < 0.5:
            a, b = b, a
        out.append({"pair_id": f"P{i:02d}", "task_id": a.task_id, "A": asdict(a), "B": asdict(b)})
    return out


def task_prompt(task_id: str) -> str:
    from macs.analysis.score import task_by_id
    return task_by_id(task_id).prompt.strip()


def save_pairs(pairs: list[dict], rating_dir: Path = RATING_DIR) -> Path:
    rating_dir = Path(rating_dir)
    rating_dir.mkdir(parents=True, exist_ok=True)
    path = rating_dir / "pairs.json"
    body = {"n_pairs": len(pairs), "pairs": pairs,
            "fingerprint": hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()[:16]}
    path.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_pairs(rating_dir: Path = RATING_DIR) -> list[dict]:
    p = Path(rating_dir) / "pairs.json"
    return json.loads(p.read_text(encoding="utf-8"))["pairs"] if p.exists() else []
