"""Is the result the same on every model? Cross-model consistency of the planned contrasts.

Each model is run and analysed on its own (results from different models are never pooled at the
session level). This script reads every per-model summary written by

    python scripts/analyze.py --all-models          # -> data/results/models/<model>/summary.json

fits the planned contrasts for each model separately, and combines each contrast across models
with a fixed-effect and a random-effects (DerSimonian-Laird) meta-analysis. It reports, per
contrast: the estimate and 95% CI on every model, the pooled estimate, Cochran's Q and I^2 (is
the effect different between models?), and how many models agree in direction.

    python scripts/model_consistency.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from macs.analysis.stats import meta_analysis, planned_contrasts  # noqa: E402
from stats_report import PRIMARY, SECONDARY, cell_rows, session_rows  # noqa: E402


def contrasts_for(summary: dict) -> dict:
    sessions, cells = session_rows(summary), cell_rows(summary)
    data = {"families_matched": cells}
    for o in ["correct_and_novel_rate", *SECONDARY]:
        data[o] = sessions
    out = {}
    for outcome, rows in data.items():
        for c in planned_contrasts(rows, outcome):
            out[c.label] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/results/models")
    ap.add_argument("--out", default="data/results/model_consistency.json")
    args = ap.parse_args()

    per_model = {}
    for f in sorted(Path(args.root).glob("*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        per_model[s.get("model") or f.parent.name] = contrasts_for(s)
    if len(per_model) < 2:
        raise SystemExit(f"need per-model summaries for at least two models in {args.root} "
                         f"(found {len(per_model)}); run: python scripts/analyze.py --all-models")
    models = list(per_model)
    labels = sorted({l for c in per_model.values() for l in c},
                    key=lambda l: (0 if any(l == f"{o}: {k}" for o, k in PRIMARY) else 1, l))
    print(f"Cross-model consistency over {len(models)} models: {', '.join(models)}\n")
    report = {"models": models, "contrasts": {}}
    for label in labels:
        rows = {m: per_model[m].get(label) for m in models}
        have = {m: c for m, c in rows.items() if c is not None}
        meta = meta_analysis([c.estimate for c in have.values()], [c.se for c in have.values()])
        report["contrasts"][label] = {"per_model": {m: c.as_dict() for m, c in have.items()}, "meta": meta}
        print(label)
        for m, c in have.items():
            print(f"   {m:<28}{c.estimate:>9.3f}   [{c.ci_low:>7.3f}, {c.ci_high:>7.3f}]  p={c.p:.3f}")
        if meta.get("k", 0) >= 2:
            lo, hi = meta["random_ci"]
            print(f"   {'pooled (random effects)':<28}{meta['random']:>9.3f}   [{lo:>7.3f}, {hi:>7.3f}]  "
                  f"p={meta['random_p']:.3f}   Q p={meta['Q_p']:.3f}  I2={meta['I2']:.0%}  "
                  f"same direction: {meta['same_sign']}/{meta['k']}")
        print()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
