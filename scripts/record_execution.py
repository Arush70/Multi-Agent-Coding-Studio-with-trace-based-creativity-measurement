"""Write the code-execution verdicts of an analysis run to data/results/execution/<model>.jsonl.

Each per-model summary (data/results/models/<model>/summary.json) lists, for every session, each
proposal's verdict (correct / Tier 3 heuristic score) in proposal order. This script pairs those
verdicts with the proposal code in the traces and writes one line per proposal, keyed by session,
proposal index and a hash of the code (see macs/analysis/execution_record.py). Later analyses then
reuse these verdicts instead of executing the code again, so the results do not depend on the
speed of the machine.

    python scripts/record_execution.py                     # from data/results/models/*/summary.json
    python scripts/record_execution.py --from other/models # from another results folder
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from macs.analysis.execution_record import RECORD_DIR, code_sha   # noqa: E402
from macs.analysis.score import implementations                   # noqa: E402
from macs.trace.writer import read_trace                          # noqa: E402


def _session_id(path: Path) -> tuple[str, Path] | None:
    with open(path, encoding="utf-8") as fh:
        first = fh.readline()
    try:
        e = json.loads(first)
    except json.JSONDecodeError:
        return None
    return (e.get("session_id"), path) if e.get("event") == "session_start" else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=str(ROOT / "data" / "results" / "models"))
    ap.add_argument("--traces", default=str(ROOT / "data" / "traces"))
    ap.add_argument("--out", default=str(RECORD_DIR))
    a = ap.parse_args()
    with ThreadPoolExecutor(16) as ex:
        paths = dict(x for x in ex.map(_session_id, sorted(Path(a.traces).glob("*.jsonl"))) if x)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    total = 0
    for summ in sorted(Path(a.src).glob("*/summary.json")):
        sessions = json.load(open(summ, encoding="utf-8"))["sessions"]
        lines = []
        for s in sessions:
            events = list(read_trace(paths[s["session_id"]]))
            codes = [c for _, c in implementations(events)]
            props = s["proposals"]
            if len(codes) != len(props):
                raise SystemExit(f"{s['session_id']}: {len(codes)} proposals in the trace, {len(props)} in {summ}")
            for i, (c, p) in enumerate(zip(codes, props)):
                lines.append(json.dumps({"session_id": s["session_id"], "i": i, "sha": code_sha(s["task_id"], c),
                                         "correct": p["correct"], "heuristic_score": p["heuristic_score"]}))
        (out / f"{summ.parent.name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        total += len(lines)
        print(f"{summ.parent.name}: {len(sessions)} sessions, {len(lines)} proposals")
    print(f"recorded {total} verdicts in {out}")


if __name__ == "__main__":
    main()
