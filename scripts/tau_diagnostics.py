"""Check the Facilitator threshold tau on the frozen reference sets, with no model calls.

For every task that has a reference set it reports:
  * tau under the original rule (25th percentile of nearest-neighbour distances among all
    reference solutions) and under the matched rule now used (25th percentile of the
    Facilitator's own window statistic on random sequences of K reference solutions);
  * the share of reference solutions with an identical twin in the representation;
  * for the studio traces of that task in --traces: how often the Facilitator would fire at the
    checks that can still change a later proposal, under the matched tau.

The proposal asks for tau to be checked on pilot tasks for a non-degenerate firing rate; this is
that check. --representation unixcoder repeats it in UniXcoder embedding space (needs the
optional torch / transformers install).

    python scripts/tau_diagnostics.py
    python scripts/tau_diagnostics.py --representation unixcoder
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macs.analysis.score import implementations  # noqa: E402
from macs.facilitator import Facilitator, tau_from_reference, tau_from_reference_sequences  # noqa: E402
from macs.metrics.reference import REFERENCE_N, REFERENCE_ROOT, load_reference_solutions  # noqa: E402
from macs.metrics.represent import ast_features, normalise  # noqa: E402
from macs.orchestrator import CONVERGENCE_WINDOW, K_PROPOSALS  # noqa: E402
from macs.trace.writer import read_trace  # noqa: E402

STUDIO = {"B", "D", "D_minus"}


def represent(codes: list[str], representation: str) -> np.ndarray:
    if representation == "unixcoder":
        from macs.metrics.represent import embed
        return embed([normalise(c) for c in codes])
    rows = []
    for c in codes:
        try:
            rows.append(ast_features(normalise(c)))
        except Exception:  # noqa: BLE001 - unparseable code has no features
            pass
    return np.vstack(rows) if rows else np.zeros((0, 26))


def window_checks(X: np.ndarray, k: int, window: int) -> list[float]:
    """The Facilitator's window metric at each check that can still change a later proposal."""
    fac = Facilitator(tau=float("inf"), window=window)
    out = []
    for i, x in enumerate(X[:k]):
        fac.observe(x)
        m = fac.window_metric()
        if m is not None and i < k - 1:
            out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-root", default=str(REFERENCE_ROOT))
    ap.add_argument("--traces", default="data/traces")
    ap.add_argument("--representation", default="ast", choices=["ast", "unixcoder"])
    ap.add_argument("--out", default="data/results/tau_diagnostics.json")
    args = ap.parse_args()

    studio_codes: dict[tuple[str, str], list[list[str]]] = {}
    for p in sorted(Path(args.traces).glob("*.jsonl")):
        events = list(read_trace(p))
        start = next((e for e in events if e["event"] == "session_start"), None)
        if start and start["condition"] in STUDIO:
            codes = [c for pid, c in implementations(events) if not str(pid).startswith("p_rev")]
            model = start.get("header", {}).get("model", "")
            studio_codes.setdefault((start["task_id"], model), []).append(codes)

    rows = []
    for ref in sorted(Path(args.reference_root).glob("*.jsonl")):
        tid = ref.stem
        codes = load_reference_solutions(tid, args.reference_root, n=REFERENCE_N)
        if len(codes) < K_PROPOSALS:
            continue
        first = json.loads(ref.read_text(encoding="utf-8").splitlines()[0])
        sessions = studio_codes.get((tid, first.get("model", "")), [])   # same model as the reference only
        X = represent(codes, args.representation)
        D = cosine_distances(X)
        np.fill_diagonal(D, np.inf)
        twins = float(np.mean(D.min(axis=1) < 1e-9))
        tau_old = tau_from_reference(X)
        tau_new = tau_from_reference_sequences(X, k=K_PROPOSALS, window=CONVERGENCE_WINDOW)
        checks = [m for s in sessions
                  for m in window_checks(represent(s, args.representation), K_PROPOSALS, CONVERGENCE_WINDOW)]
        fire = float(np.mean([m < tau_new for m in checks])) if checks and tau_new > 0 else None
        rows.append({"task_id": tid, "n_reference": len(X), "share_with_identical_twin": round(twins, 3),
                     "tau_old_rule": round(tau_old, 5), "tau_matched_rule": round(tau_new, 5),
                     "studio_sessions": len(sessions), "studio_checks": len(checks),
                     "studio_fire_rate": None if fire is None else round(fire, 3)})

    print(f"representation: {args.representation}   K={K_PROPOSALS}  window={CONVERGENCE_WINDOW}\n")
    print(f"{'task':<36}{'n':>4}{'twins':>7}{'tau_old':>10}{'tau_new':>10}{'sessions':>10}{'fire':>7}")
    for r in rows:
        fire = "-" if r["studio_fire_rate"] is None else f"{r['studio_fire_rate']:.2f}"
        print(f"{r['task_id']:<36}{r['n_reference']:>4}{r['share_with_identical_twin']:>7.2f}"
              f"{r['tau_old_rule']:>10.4f}{r['tau_matched_rule']:>10.4f}{r['studio_sessions']:>10}{fire:>7}")
    if not rows:
        print("no reference sets found; build them with scripts/build_reference.py")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"representation": args.representation, "k": K_PROPOSALS,
                               "window": CONVERGENCE_WINDOW, "tasks": rows}, indent=1))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
