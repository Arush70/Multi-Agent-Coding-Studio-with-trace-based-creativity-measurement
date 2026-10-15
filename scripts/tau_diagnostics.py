"""Check the Facilitator threshold tau on the frozen reference sets, with no model calls.

For every task that has a reference set it reports:
  * tau under the original rule (25th percentile of nearest-neighbour distances among all
    reference solutions) and under the matched rule now used (25th percentile of the
    Facilitator's own window statistic on random sequences of K reference solutions);
  * the share of reference solutions with an identical twin in the representation;
  * for the studio traces of that task in --traces (by default D_minus only, where the
    Facilitator is off, so its own instructions do not shape the proposals): how often it would
    fire at the checks that can still change a later proposal under the matched tau, and where
    those window values fall in the reference distribution (percentile rank).

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

FIRE_LOW, FIRE_HIGH = 0.10, 0.90     # pre-registered band for a non-degenerate firing rate
PERCENTILES = (25, 50, 75)           # tried in this order by the pre-registered rule


def represent(codes: list[str], representation: str) -> np.ndarray:
    if representation == "unixcoder":
        from macs.metrics.represent import embed
        normed = []
        for c in codes:
            try:
                normed.append(normalise(c))
            except Exception:  # noqa: BLE001 - unparseable code is skipped, as for AST
                pass
        return embed(normed) if normed else np.zeros((0, 768))
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


def reference_windows(X: np.ndarray, k: int, window: int, n_draws: int = 2000, seed: int = 0) -> np.ndarray:
    """The window statistic on random k-sequences of reference solutions (as in the tau rule)."""
    rng = np.random.default_rng(seed)
    return np.array([m for _ in range(n_draws)
                     for m in window_checks(X[rng.choice(len(X), size=k, replace=False)], k, window)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-root", default=str(REFERENCE_ROOT))
    ap.add_argument("--traces", default="data/traces")
    ap.add_argument("--representation", default="ast", choices=["ast", "unixcoder"])
    ap.add_argument("--conditions", default="D_minus",
                    help="studio conditions whose traces are checked (comma-separated)")
    ap.add_argument("--out", default=None, help="default: data/results/tau_diagnostics_<representation>.json")
    args = ap.parse_args()

    studio_codes: dict[tuple[str, str], list[list[str]]] = {}
    for p in sorted(Path(args.traces).glob("*.jsonl")):
        events = list(read_trace(p))
        start = next((e for e in events if e["event"] == "session_start"), None)
        end = str(events[-1].get("content", "")) if events and events[-1]["event"] == "session_end" else ""
        if not end.startswith(("completed", "stopped")):
            continue                     # a session that crashed is not a studio sample
        if start and start["condition"] in set(args.conditions.split(",")):
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
        ref_w = reference_windows(X, K_PROPOSALS, CONVERGENCE_WINDOW)
        checks = [m for s in sessions
                  for m in window_checks(represent(s, args.representation), K_PROPOSALS, CONVERGENCE_WINDOW)]
        fire = float(np.mean([m < tau_new for m in checks])) if checks and tau_new > 0 else None
        rank = float(np.median([np.mean(ref_w < m) for m in checks])) if checks else None
        by_pct = {}
        for pc in PERCENTILES:
            t = tau_from_reference_sequences(X, k=K_PROPOSALS, window=CONVERGENCE_WINDOW, percentile=pc)
            fr = float(np.mean([m < t for m in checks])) if checks and t > 0 else None
            by_pct[str(pc)] = {"tau": round(t, 5), "fire_rate": None if fr is None else round(fr, 3),
                               "non_degenerate": None if fr is None else FIRE_LOW <= fr <= FIRE_HIGH}
        rows.append({"task_id": tid, "n_reference": len(X), "share_with_identical_twin": round(twins, 3),
                     "tau_old_rule": round(tau_old, 5), "tau_matched_rule": round(tau_new, 5),
                     "studio_sessions": len(sessions), "studio_checks": len(checks),
                     "studio_fire_rate": None if fire is None else round(fire, 3),
                     "studio_median_percentile_rank": None if rank is None else round(rank, 3),
                     "non_degenerate": None if fire is None else FIRE_LOW <= fire <= FIRE_HIGH,
                     "by_percentile": by_pct})

    print(f"representation: {args.representation}   K={K_PROPOSALS}  window={CONVERGENCE_WINDOW}\n")
    print(f"studio traces: {args.conditions}; non-degenerate firing = {FIRE_LOW:.0%}-{FIRE_HIGH:.0%} of checks\n")
    print(f"{'task':<36}{'n':>4}{'twins':>7}{'tau_old':>10}{'tau_new':>10}{'sessions':>9}{'checks':>8}"
          f"{'fire':>7}{'rank':>7}")
    for r in rows:
        fire = "-" if r["studio_fire_rate"] is None else f"{r['studio_fire_rate']:.2f}"
        rank = "-" if r["studio_median_percentile_rank"] is None else f"{r['studio_median_percentile_rank']:.2f}"
        print(f"{r['task_id']:<36}{r['n_reference']:>4}{r['share_with_identical_twin']:>7.2f}"
              f"{r['tau_old_rule']:>10.4f}{r['tau_matched_rule']:>10.4f}{r['studio_sessions']:>9}"
              f"{r['studio_checks']:>8}{fire:>7}{rank:>7}")
    print(f"\nwould-fire rate by percentile ({FIRE_LOW:.0%}-{FIRE_HIGH:.0%} counts as non-degenerate):")
    print(f"{'task':<36}" + "".join(f"{'p' + str(pc):>10}" for pc in PERCENTILES))
    for r in rows:
        cells = []
        for pc in PERCENTILES:
            fr = r["by_percentile"][str(pc)]["fire_rate"]
            ok = r["by_percentile"][str(pc)]["non_degenerate"]
            cells.append("-" if fr is None else f"{fr:.2f}{'*' if ok else ' '}")
        print(f"{r['task_id']:<36}" + "".join(f"{c:>10}" for c in cells))
    print("(* = inside the band)")
    print("\nrank = median percentile of the studio's window values within the reference's own window "
          "values;\n       the matched rule fires below the 25th percentile.")
    if not rows:
        print("no reference sets found; build them with scripts/build_reference.py")
    out = Path(args.out or f"data/results/tau_diagnostics_{args.representation}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"representation": args.representation, "k": K_PROPOSALS,
                               "window": CONVERGENCE_WINDOW, "conditions": args.conditions,
                               "fire_band": [FIRE_LOW, FIRE_HIGH], "tasks": rows}, indent=1), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
