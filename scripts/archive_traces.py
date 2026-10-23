"""Move traces that the current orchestrator no longer analyses into data/archive/<name>/.

Nothing is deleted. Traces from versions older than COMPATIBLE_SINCE (for example the 2-learner,
4096-context runs before orchestrator 0.4.0) go to data/archive/<name>/traces, and the analysis
results made from them go to data/archive/<name>/results, so the dashboard and analysis start
clean while the old run stays available (it is reported as a pilot).

    python scripts/archive_traces.py --name 2026-10-23_two_learner_ctx4096            # see what would move
    python scripts/archive_traces.py --name 2026-10-23_two_learner_ctx4096 --apply    # move it
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from macs.orchestrator import COMPATIBLE_SINCE, ORCHESTRATOR_VERSION, is_comparable  # noqa: E402


def header(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as f:
            first = json.loads(f.readline())
        return {"condition": first.get("condition"), **first.get("header", {})}
    except (OSError, ValueError):
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="archive folder name, e.g. 2026-10-23_two_learner_ctx4096")
    ap.add_argument("--traces", default="data/traces")
    ap.add_argument("--apply", action="store_true", help="move the files (default: only list them)")
    ap.add_argument("--keep-results", action="store_true", help="leave data/results in place")
    args = ap.parse_args()

    dest = ROOT / "data" / "archive" / args.name
    old = []
    for p in sorted((ROOT / args.traces).glob("*.jsonl")):
        h = header(p)
        if not is_comparable(h.get("orchestrator_version")):
            old.append((p, h))
    by = Counter((h.get("model", "?"), h.get("condition", "?")) for _, h in old)
    print(f"{len(old)} trace(s) older than {COMPATIBLE_SINCE} (current {ORCHESTRATOR_VERSION}) -> {dest / 'traces'}")
    for (model, cond), n in sorted(by.items()):
        print(f"  {model:<24} {cond:<8} {n}")
    extra = [ROOT / "data" / "jobs" / "smoke_traces"]
    if not args.keep_results:
        # tau diagnostics describe the reference sets, which stay valid, so they are kept
        extra += [p for p in (ROOT / "data" / "results").glob("*") if p.is_file() and not p.name.startswith("tau_")]
    extra = [p for p in extra if p.exists()]
    for p in extra:
        print(f"  also: {p.relative_to(ROOT)}")
    if not args.apply:
        print("\n(dry run; add --apply to move them)")
        return
    (dest / "traces").mkdir(parents=True, exist_ok=True)
    (dest / "results").mkdir(parents=True, exist_ok=True)
    for p, _ in old:
        shutil.move(str(p), str(dest / "traces" / p.name))
    for p in extra:
        target = dest / ("results" if p.parent.name == "results" else "") / p.name
        shutil.move(str(p), str(target))
    (dest / "README.txt").write_text(
        f"Archived {len(old)} traces older than orchestrator {COMPATIBLE_SINCE}.\n"
        "Reason: from 0.4.0 the second learner level is 6 (plus 4 for the curve) instead of 2, and the\n"
        "model server must give an 8192-token context (the 4096 context cut some Critic prompts short).\n"
        "These traces are kept as the 2-learner pilot and are not part of the main analysis.\n"
        + "".join(f"{m}\t{c}\t{n}\n" for (m, c), n in sorted(by.items())), encoding="utf-8")
    print(f"moved to {dest}")


if __name__ == "__main__":
    main()
