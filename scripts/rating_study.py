"""The rating study (SC4), end to end.

    python scripts/rating_study.py build              # 60 pairs of correct solutions -> data/rating/pairs.json
    python scripts/rating_study.py export             # rater booklet (HTML) + Microsoft Forms Quick-import (.docx)
    python scripts/rating_study.py import FILE.xlsx   # the Form's Excel export -> data/rating/responses.json
    python scripts/rating_study.py judge              # independent AI judge (profile 'deepseek') -> judge.json
    python scripts/rating_study.py analyse            # Bradley-Terry, kappa, Spearman -> results.json

Pairs are built once and then frozen: `build` refuses to overwrite data/rating/pairs.json once
responses exist, so the raters and the analysis always see the same pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macs.rating import analysis, forms, labelling as L, pairs as P  # noqa: E402

DIR = P.RATING_DIR


def _prompts(pairs, items=()):
    return {t: P.task_prompt(t) for t in {p["task_id"] for p in pairs} | {i["task_id"] for i in items}}


def cmd_build(a):
    if (DIR / "responses.json").exists() and not a.force:
        raise SystemExit("responses already imported; the pairs are frozen (use --force only to start over)")
    sols = P.collect_solutions(a.traces, model=a.model, backend=a.sandbox)
    pairs = P.build_pairs(sols, n_pairs=a.n, seed=a.seed)
    path = P.save_pairs(pairs, DIR)
    tasks = sorted({p["task_id"] for p in pairs})
    conds = sorted({p[s]["condition"] for p in pairs for s in "AB"})
    print(f"{len(pairs)} pairs from {len(tasks)} task(s); solutions from conditions {', '.join(conds)} -> {path}")
    if len(pairs) < a.n:
        print(f"[warn] only {len(pairs)} pairs could be formed; run more sessions (more seeds or tasks) first")
    from macs.analysis.aggregate import resolve_model
    items = L.build_items(a.traces, model=resolve_model(a.traces, a.model))
    print(f"{len(items)} Part B labelling items from {len({i['task_id'] for i in items})} task(s) -> {L.save_items(items, DIR)}")


def cmd_export(a):
    pairs = P.load_pairs(DIR)
    if not pairs:
        raise SystemExit("no pairs yet: run `build` first")
    items = L.load_items(DIR)
    b = forms.export_booklet(pairs, _prompts(pairs, items), DIR / "rating_booklet.html", items)
    f = forms.export_form(pairs, DIR / "form_quick_import.docx", items)
    print(f"booklet: {b}  (open in a browser, print to PDF for the raters)\nform:    {f}  "
          "(Microsoft Forms > New > Quick import; add the consent page from the ethics consent form first)")


def cmd_import(a):
    long, exp = forms.read_responses(Path(a.file))
    labels = L.read_label_responses(forms._rows(Path(a.file)))
    (DIR / "responses.json").write_text(json.dumps({"responses": long, "experience": exp, "labels": labels},
                                                   indent=1), encoding="utf-8")
    print(f"Part B: {len(labels)} labels")
    raters = sorted({r["rater"] for r in long})
    print(f"{len(long)} answers from {len(raters)} rater(s): {', '.join(raters)} -> {DIR / 'responses.json'}")


def cmd_judge(a):
    from macs.llm.client import LLMClient
    pairs = P.load_pairs(DIR)
    if not pairs:
        raise SystemExit("no pairs yet: run `build` first")
    from macs.rating.judge import judge_pairs
    client = LLMClient.from_config(profile=a.profile)
    from macs.rating.judge import judge_labels
    items = L.load_items(DIR)
    resp, st = judge_pairs(pairs, _prompts(pairs, items), client)
    labels, lst = judge_labels(items, _prompts(pairs, items), client) if items else ([], {})
    (DIR / "judge.json").write_text(json.dumps({"model": client.profile.model, "responses": resp, "stats": st,
                                               "labels": labels, "label_stats": lst}, indent=1), encoding="utf-8")
    print(f"judge {client.profile.model}: pairs {st}; Part B labels {lst}")


def cmd_analyse(a):
    pairs = P.load_pairs(DIR)
    res = {}
    if (DIR / "responses.json").exists():
        data = json.loads((DIR / "responses.json").read_text(encoding="utf-8"))
        human = data["responses"]
        res["humans"] = analysis.analyse(pairs, human)
        if data.get("labels"):
            res["sc3_labels"] = L.analyse_labels(L.load_items(DIR), data["labels"])
            print(f"\nSC3 Part B: {res['sc3_labels']}\n  -> data/calibration/labels.jsonl (majority); "
                  "now run: python scripts/calibrate_clustering.py")
    if (DIR / "judge.json").exists():
        judge = json.loads((DIR / "judge.json").read_text(encoding="utf-8"))
        res["judge"] = analysis.analyse(pairs, judge["responses"])
        res["judge"]["stats"] = judge["stats"]
        if judge.get("labels"):
            # the AI rater's labels go to their own folder, never into the human labels file
            res["sc3_labels_judge"] = L.analyse_labels(L.load_items(DIR), judge["labels"],
                                                       calib_dir=Path("data/calibration/judge"))
        if "humans" in res:
            res["judge_vs_humans"] = analysis.judge_agreement(human, judge["responses"])
    if not res:
        raise SystemExit("nothing to analyse: import responses or run the judge first")
    for who, r in res.items():
        if who in ("judge_vs_humans", "sc3_labels", "sc3_labels_judge"):
            if who == "judge_vs_humans":
                print(f"\njudge vs human majority: {r}")
            continue
        print(f"\n== {who} ({', '.join(r['raters'])})")
        for q, v in r["questions"].items():
            print(f"  {q:<9} rho={v['spearman_rho']} CI={v['rho_ci']}  kappa={v['mean_cohen_kappa']}  "
                  f"fleiss={v['fleiss_kappa']}  majority picks more novel={v['majority_picks_more_novel']} "
                  f"(n={v['n_majority_pairs']})")
    out = DIR / "results.json"
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\nwritten: {out}  (SC4 targets: Spearman rho >= 0.50, Cohen's kappa >= 0.60)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--n", type=int, default=P.N_PAIRS); b.add_argument("--seed", type=int, default=2026)
    b.add_argument("--traces", default="data/traces"); b.add_argument("--model", default=None)
    b.add_argument("--sandbox", default="subprocess", choices=["subprocess", "docker"]); b.add_argument("--force", action="store_true")
    sub.add_parser("export")
    i = sub.add_parser("import"); i.add_argument("file")
    j = sub.add_parser("judge"); j.add_argument("--profile", default="deepseek")
    sub.add_parser("analyse")
    a = ap.parse_args()
    {"build": cmd_build, "export": cmd_export, "import": cmd_import, "judge": cmd_judge, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
