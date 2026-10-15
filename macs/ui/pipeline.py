"""Pipeline helpers for the dashboard (no Streamlit here, so they can be tested).

References: per-task status, archiving old reference files.
Multi-run: profiles from config/models.yaml, seed specs, how many sessions a run would add.
SC3 labelling: a blind pool of solutions to label, label files per rater, agreement between raters.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "data" / "reference"
REFERENCE_ARCHIVE = ROOT / "data" / "reference_archive"
TRACES_DIR = ROOT / "data" / "traces"
CONFIG = ROOT / "config" / "models.yaml"
CALIB_DIR = ROOT / "data" / "calibration"
OTHER, UNSURE = "other", "unsure"


# ------------------------------------------------------------------ model profiles

def list_profiles(config: Path = CONFIG) -> list[dict]:
    """Profiles in config/models.yaml without their API keys."""
    import yaml
    try:
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    except OSError:
        return []
    out = []
    for name, p in (cfg.get("profiles") or {}).items():
        out.append({"name": name, "model": p.get("model", ""), "revision": p.get("revision", ""),
                    "base_url": p.get("base_url", ""), "extra_body": p.get("extra_body") or {},
                    "default": name == cfg.get("default_profile")})
    return out


# ------------------------------------------------------------------ references

def _reference_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def reference_table(tasks, root: Path = REFERENCE_DIR, model: str | None = None, target: int = 100) -> list[dict]:
    """One row per task: samples present, which model made them, and whether it is usable."""
    out = []
    for t in tasks:
        rows = _reference_rows(Path(root) / f"{t.id}.jsonl")
        models = sorted({f"{r.get('model', '?')}@{r.get('revision', '')}" for r in rows})
        names = {r.get("model") for r in rows}
        if not rows:
            state = "missing"
        elif model is not None and names != {model}:
            state = "other model"
        elif len(rows) < target:
            state = "incomplete"
        else:
            state = "ready"
        out.append({"task": t.id, "tier": t.tier, "samples": len(rows), "target": target,
                     "model": ", ".join(models) or "-", "state": state})
    return out


def archive_references(task_ids: list[str] | None = None, root: Path = REFERENCE_DIR,
                       archive_root: Path = REFERENCE_ARCHIVE, stamp: str | None = None) -> tuple[Path | None, int]:
    """Move reference files (all, or the given tasks) into archive_root/<stamp>/. Never deletes."""
    files = sorted(Path(root).glob("*.jsonl"))
    if task_ids is not None:
        wanted = set(task_ids)
        files = [f for f in files if f.stem in wanted]
    if not files:
        return None, 0
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    dest = Path(archive_root) / stamp
    n = 1
    while dest.exists():
        dest = Path(archive_root) / f"{stamp}-{n}"
        n += 1
    dest.mkdir(parents=True)
    for f in files:
        f.replace(dest / f.name)
    return dest, len(files)


# ------------------------------------------------------------------ multi-run

def seeds_spec(first: int, count: int) -> str:
    """first=1, count=6 -> '1-6'."""
    if count < 1:
        raise ValueError("at least one seed")
    return str(first) if count == 1 else f"{first}-{first + count - 1}"


def run_plan(task_ids: list[str], conditions: list[str], seeds: list[int], model: str | None,
             traces_dir: Path = TRACES_DIR) -> dict:
    """How many sessions a multi-run covers, and how many are already done with this model."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_experiment import done_cells
    done = done_cells(Path(traces_dir), model=model)
    cells = [(t, c, s) for t in task_ids for c in conditions for s in seeds]
    already = sum(1 for cell in cells if cell in done)
    return {"total": len(cells), "done": already, "to_run": len(cells) - already}


# ------------------------------------------------------------------ SC3 labelling

def _key(task_id: str, session: str, proposal_id: str) -> str:
    return f"{task_id}|{session}|{proposal_id}"


def label_pool(traces_dir: Path = TRACES_DIR, per_task: int = 40, seed: int = 0) -> list[dict]:
    """A blind labelling pool from the current traces: for each task, up to `per_task` solutions
    from all conditions, in a fixed shuffled order so the condition cannot be read off the order.
    Only traces from the current orchestrator and the newest model are used."""
    from macs.analysis.score import implementations
    from macs.orchestrator import ORCHESTRATOR_VERSION
    from macs.trace.writer import read_trace
    loaded = []
    for p in sorted(Path(traces_dir).glob("*.jsonl")):
        try:
            events = list(read_trace(p))
        except (OSError, json.JSONDecodeError):
            continue
        if not events or events[0].get("event") != "session_start":
            continue
        h = events[0].get("header", {})
        if h.get("orchestrator_version") != ORCHESTRATOR_VERSION:
            continue
        loaded.append((p, events, h))
    if not loaded:
        return []
    newest = max(loaded, key=lambda x: str(x[1][0].get("ts", "")))[2].get("model")
    by_task: dict[str, list[dict]] = {}
    for p, events, h in loaded:
        if h.get("model") != newest:
            continue
        tid = events[0]["task_id"]
        for pid, code in implementations(events):
            if code.strip():
                by_task.setdefault(tid, []).append({"key": _key(tid, p.stem, pid), "task_id": tid,
                                                     "session": p.stem, "proposal_id": pid, "code": code})
    pool = []
    for tid, rows in sorted(by_task.items()):
        rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['key']}".encode()).hexdigest())
        pool.extend(rows[:per_task])
    return pool


def labels_path(rater: str, calib_dir: Path = CALIB_DIR) -> Path:
    """Rater 1's file is the one scripts/calibrate_clustering.py reads (labels.jsonl)."""
    return Path(calib_dir) / ("labels.jsonl" if rater == "rater1" else f"labels_{rater}.jsonl")


def load_labels(path: Path) -> dict[str, dict]:
    out = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[_key(r["task_id"], r["session"], r["proposal_id"])] = r
    return out


def save_label(path: Path, item: dict, family: str) -> None:
    """Record (or change) the family for one pooled solution; the file stays one row per solution."""
    labels = load_labels(path)
    labels[item["key"]] = {"task_id": item["task_id"], "session": item["session"],
                           "proposal_id": item["proposal_id"], "family": family, "code": item["code"]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels.values()), encoding="utf-8")
    tmp.replace(path)


def rater_agreement(path_a: Path, path_b: Path) -> dict:
    """Cohen's kappa between two raters on the solutions both labelled ('unsure' excluded)."""
    a, b = load_labels(path_a), load_labels(path_b)
    both = [k for k in a if k in b and a[k]["family"] != UNSURE and b[k]["family"] != UNSURE]
    if len(both) < 2:
        return {"n": len(both), "kappa": None, "agreement": None}
    from sklearn.metrics import cohen_kappa_score
    ya, yb = [a[k]["family"] for k in both], [b[k]["family"] for k in both]
    agree = sum(x == y for x, y in zip(ya, yb)) / len(both)
    kappa = float(cohen_kappa_score(ya, yb)) if len(set(ya) | set(yb)) > 1 else 1.0
    return {"n": len(both), "kappa": round(kappa, 3), "agreement": round(agree, 3)}
