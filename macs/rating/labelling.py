"""SC3 blind labelling by independent raters (Part B of the rater form, amended 2026-10-25).

Raters are classmates or acquaintances of the researcher, never the researcher or the
supervisor. For a sample of solutions they choose which of the task's pre-registered approaches
(`known_families`, written before any model output) each solution uses, or "Something else".
The majority label per solution is the human reference: the automatic clustering is compared with
it by ARI (scripts/calibrate_clustering.py reads data/calibration/labels.jsonl), and raters'
agreement with each other is reported as Fleiss' kappa.

Task choice is fixed by rule, not by looking at results: the main-bank Tier 1 and Tier 2 tasks
with four known families, at most three per tier, in id order (Tier 3 code is long and needs
domain knowledge). Solutions: distinct (normalised) implementations from completed sessions of
the main model, all conditions, in a seeded hashed order; raters see the blind view only
(comments and docstrings removed), never the condition.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

RATING_DIR = Path("data/rating")
ITEMS_FILE = "label_items.json"
PER_TASK = 8
PER_TIER = 3
OTHER = "Something else / none of these"
Q_LABEL = "Which approach does this solution use?"
_ITEM_COL = re.compile(r"Item\s*L(\d+)", re.I)


def family_descriptions(task_id: str) -> list[tuple[str, str]]:
    """[(family, plain description)] from the task's YAML (the comment after each family)."""
    from macs.tasks.bank import load_bank
    task = next(t for t in load_bank() if t.id == task_id)
    text = ""
    for p in Path("tasks").rglob("task.yaml"):
        if p.parent.name == task_id:
            text = p.read_text(encoding="utf-8")
            break
    out = []
    for fam in task.known_families:
        m = re.search(rf"-\s*{re.escape(fam)}\s*#\s*(.+)", text)
        out.append((fam, m.group(1).strip() if m else fam.replace("_", " ")))
    return out


def option_text(fam: str, desc: str) -> str:
    return f"{fam.replace('_', ' ')}: {desc}"


def choose_tasks() -> list[str]:
    from macs.tasks.bank import load_bank
    out = []
    for tier in (1, 2):
        ids = sorted(t.id for t in load_bank() if t.task_set == "main" and t.tier == tier
                     and len(t.known_families) == 4)
        out += ids[:PER_TIER]
    return out


def build_items(traces_dir="data/traces", *, model: str, per_task: int = PER_TASK, seed: int = 2026) -> list[dict]:
    from macs.analysis.score import implementations
    from macs.metrics.represent import normalise
    from macs.orchestrator import is_comparable
    from macs.trace.writer import read_trace
    from macs.ui.pipeline import blind_view
    tasks = choose_tasks()
    pool: dict[str, list[dict]] = {t: [] for t in tasks}
    seen = set()
    for p in sorted(Path(traces_dir).glob("*.jsonl")):
        events = list(read_trace(p))
        if not events or events[0].get("event") != "session_start":
            continue
        start, h = events[0], events[0].get("header", {})
        if h.get("model") != model or start["task_id"] not in pool or not is_comparable(h.get("orchestrator_version")):
            continue
        if not str(events[-1].get("content", "")).startswith(("completed", "stopped")):
            continue
        for pid, code in implementations(events):
            try:
                key = (start["task_id"], normalise(code))
            except Exception:  # noqa: BLE001 - unparseable code is not labelled
                continue
            if key in seen:
                continue
            seen.add(key)
            pool[start["task_id"]].append({"task_id": start["task_id"], "session": p.stem, "proposal_id": pid,
                                           "condition": start["condition"], "display": blind_view(code)})
    items = []
    for t in tasks:
        rows = sorted(pool[t], key=lambda r: hashlib.sha256(f"{seed}:{r['session']}:{r['proposal_id']}".encode()).hexdigest())
        items += rows[:per_task]
    order = sorted(range(len(items)), key=lambda i: hashlib.sha256(f"{seed}:order:{i}".encode()).hexdigest())
    items = [items[i] for i in order]                     # tasks interleaved, so no block reveals a pattern
    for n, it in enumerate(items, start=1):
        it["item_id"] = f"L{n:02d}"
        it["options"] = [option_text(f, d) for f, d in family_descriptions(it["task_id"])] + [OTHER]
    return items


def save_items(items: list[dict], rating_dir: Path = RATING_DIR) -> Path:
    path = Path(rating_dir) / ITEMS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=1), encoding="utf-8")
    return path


def load_items(rating_dir: Path = RATING_DIR) -> list[dict]:
    path = Path(rating_dir) / ITEMS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def question_text(item_id: str) -> str:
    return f"Item {item_id}: {Q_LABEL}"


def read_label_responses(rows: list[dict]) -> list[dict]:
    """[{rater, item_id, choice}] from the Form export rows (columns 'Item Lxx: ...')."""
    out = []
    for i, row in enumerate(rows, start=1):
        rater = next((v.strip() for k, v in row.items() if "rater code" in k.lower() and v), "") or f"rater{i}"
        for k, v in row.items():
            m = _ITEM_COL.search(k)
            if m and str(v).strip():
                out.append({"rater": rater, "item_id": f"L{int(m.group(1)):02d}", "choice": str(v).strip()})
    return out


def _family(choice: str, item: dict) -> str:
    """Option text -> family name ('other' for anything else)."""
    for opt in item["options"]:
        if choice.strip() == opt and opt != OTHER:
            return opt.split(":", 1)[0].strip().replace(" ", "_")
    return "other"


def analyse_labels(items: list[dict], responses: list[dict], calib_dir: Path = Path("data/calibration")) -> dict:
    """Fleiss' kappa across raters, the majority label per item, and the labels files that
    scripts/calibrate_clustering.py reads (labels.jsonl = majority; labels_<rater>.jsonl each)."""
    from macs.rating.analysis import fleiss_kappa
    by_id = {it["item_id"]: it for it in items}
    votes: dict[str, list[str]] = {}
    per_rater: dict[str, dict[str, str]] = {}
    for r in responses:
        it = by_id.get(r["item_id"])
        if it is None:
            continue
        fam = _family(r["choice"], it)
        votes.setdefault(r["item_id"], []).append(fam)
        per_rater.setdefault(r["rater"], {})[r["item_id"]] = fam
    cats = sorted({f for v in votes.values() for f in v})
    n_r = Counter(len(v) for v in votes.values()).most_common(1)[0][0] if votes else 0
    table = [[Counter(v)[c] for c in cats] for v in votes.values() if len(v) == n_r]
    kappa = fleiss_kappa(table) if n_r >= 2 and table else None
    majority, ties = {}, 0
    for iid, v in votes.items():
        top = Counter(v).most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            ties += 1
            continue                                      # no majority: left out of the ARI
        majority[iid] = top[0][0]
    calib_dir = Path(calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)

    def write(path: Path, labels: dict[str, str]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for iid, fam in sorted(labels.items()):
                it = by_id[iid]
                fh.write(json.dumps({"task_id": it["task_id"], "session": it["session"],
                                     "proposal_id": it["proposal_id"], "family": fam, "item_id": iid}) + "\n")
    write(calib_dir / "labels.jsonl", majority)
    for rater, labels in per_rater.items():
        write(calib_dir / f"labels_{re.sub(r'[^A-Za-z0-9_-]', '_', rater)}.jsonl", labels)
    return {"n_items": len(items), "n_labelled": len(votes), "raters": sorted(per_rater),
            "fleiss_kappa": kappa, "raters_per_item": n_r, "majority_items": len(majority), "ties": ties,
            "label_counts": dict(Counter(majority.values()))}
