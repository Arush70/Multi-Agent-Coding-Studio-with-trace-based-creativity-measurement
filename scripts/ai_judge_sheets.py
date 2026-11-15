"""Part A pairs for an AI judge used through a chat interface (exploratory, 2026-11-15).

export  writes data/rating/ai_judge/PartA_order1.{html,txt} and PartA_order2.{html,txt}: the same
        60 pairs as the human raters saw (task, two solutions, the two questions); in order 2 the
        two solutions of every pair are swapped (A <-> B).
import  reads the judge's two answer files (lines "P01: A B" = original, creative) and keeps a choice
        only when both orders pick the same solution; writes data/rating/ai_judge/<judge>.json and
        prints the same analysis as for the human raters, plus agreement with the human majority.

Each order is given in a fresh chat with no other context.
"""
from __future__ import annotations

import argparse, html, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from macs.rating import analysis, pairs as P  # noqa: E402

OUT = P.RATING_DIR / "ai_judge"
Q1 = "Which of these two correct solutions takes the more original approach?"
Q2 = "Which is the more creative solution?"

INSTR = f"""INSTRUCTIONS FOR THE RATER (please read before answering)

Your role
You are an independent rater in a university research study about program code. Please work only
from this document; do not search the web and do not use anything from earlier conversations.

What you see
There are 60 pairs, numbered P01 to P60. Each pair shows a short programming problem and two
Python solutions to it, labelled A and B. Both solutions are correct: they pass the problem's tests.

Your task
For each pair, answer two questions by choosing A or B:
  1. {Q1}
  2. {Q2}
  - Judge the approach: the idea, method or data structure used to solve the problem.
  - Ignore surface features: variable names, comments, layout, and length on its own.
  - If the two seem equally original (or creative), choose the one you lean towards.
  - You do not need to run the code. You are not told how the solutions were produced; please do
    not guess.

How to answer
Answer every pair, from P01 to P60. Reply with exactly 60 lines and nothing else: no explanations,
no headings, no code blocks. Each line has the pair number, a colon, a space, your answer to
question 1, a space, and your answer to question 2, like this:
P01: A B
P02: B B
...
P60: A A"""

END = """END OF PAIRS. Now give your answer: exactly 60 lines, P01 to P60, each in the form
"P01: A B" (question 1, then question 2), with no other text."""


def sides(p, order):
    return (p["A"], p["B"]) if order == 1 else (p["B"], p["A"])


def export(_a):
    pairs = P.load_pairs(P.RATING_DIR)
    prompts = {t: P.task_prompt(t) for t in {p["task_id"] for p in pairs}}
    OUT.mkdir(parents=True, exist_ok=True)
    e = html.escape
    for order in (1, 2):
        txt = [f"Pairwise comparison - order {order} of 2 - {len(pairs)} pairs", "", INSTR, ""]
        hb = []
        for p in pairs:
            a, b = sides(p, order)
            pid = p["pair_id"]
            txt += ["=" * 70, f"Pair {pid}", "", "Problem:", prompts[p["task_id"]].strip(), "",
                    "Solution A:", a["display"], "", "Solution B:", b["display"], "",
                    f"1. {Q1}  (A or B)", f"2. {Q2}  (A or B)", ""]
            hb.append(f"<div class='it'><h2>Pair {pid}</h2><div class='task'>{e(prompts[p['task_id']].strip())}</div>"
                      f"<div class='lab'>Solution A</div><pre>{e(a['display'])}</pre>"
                      f"<div class='lab'>Solution B</div><pre>{e(b['display'])}</pre>"
                      f"<ol><li>{e(Q1)} (A or B)</li><li>{e(Q2)} (A or B)</li></ol></div>")
        txt += ["=" * 70, END]
        (OUT / f"PartA_order{order}.txt").write_text("\n".join(txt), encoding="utf-8")
        (OUT / f"PartA_order{order}.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Pairwise comparison, order {order}</title><style>
@page{{size:A4;margin:14mm}} body{{font-family:Arial,sans-serif;font-size:10pt;color:#111}}
h1{{font-size:15pt}} h2{{font-size:11.5pt;margin:0 0 4pt}}
.ins{{white-space:pre;background:#f2f4f7;padding:8pt;font-family:Consolas,monospace;font-size:8pt;line-height:1.35}}
.it{{page-break-inside:avoid;border-top:1px solid #ccc;padding-top:6pt;margin-top:8pt}}
.task{{white-space:pre-wrap;font-size:9pt;color:#333;margin-bottom:4pt}} .lab{{font-weight:bold;font-size:9pt;margin-top:4pt}}
pre{{font-family:Consolas,monospace;font-size:8.3pt;background:#f6f7f9;padding:5pt;white-space:pre-wrap;margin:2pt 0}}
ol{{margin:3pt 0 0 0}}</style></head><body>
<h1>Pairwise comparison: order {order} of 2 ({len(pairs)} pairs)</h1>
<div class="ins">{e(INSTR)}</div>{''.join(hb)}<div class="ins" style="margin-top:10pt">{e(END)}</div></body></html>""",
                                                         encoding="utf-8")
    print("written to", OUT)


def parse(path):
    got = {}
    for m in re.finditer(r"\bP(\d{2})\s*[:.)-]?\s+([AB])\b[\s,/;]*([AB])\b", Path(path).read_text(encoding="utf-8")):
        got[f"P{m.group(1)}"] = (m.group(2), m.group(3))
    return got


def imp(a):
    pairs = P.load_pairs(P.RATING_DIR)
    r1, r2 = parse(a.order1), parse(a.order2)
    flip = {"A": "B", "B": "A"}
    resp, stats = [], {}
    for k, q in enumerate(("original", "creative")):
        ok = 0
        for p in pairs:
            pid = p["pair_id"]
            if pid not in r1 or pid not in r2:
                continue
            c1, c2 = r1[pid][k], flip[r2[pid][k]]       # order 2 answers mapped back to the original A/B
            if c1 == c2:
                ok += 1
                resp.append({"rater": a.judge, "pair_id": pid, "question": q, "choice": c1})
        stats[q] = {"consistent": ok, "of": len(pairs)}
    OUT.mkdir(parents=True, exist_ok=True)
    res = analysis.analyse(pairs, resp)
    human = json.loads((P.RATING_DIR / "responses.json").read_text(encoding="utf-8"))["responses"]
    agree = analysis.judge_agreement(human, resp)
    (OUT / f"{a.judge}.json").write_text(json.dumps({"judge": a.judge, "answered": [len(r1), len(r2)],
                                                     "stats": stats, "responses": resp, "analysis": res,
                                                     "vs_human_majority": agree}, indent=1, default=float),
                                         encoding="utf-8")
    print(f"{a.judge}: answered {len(r1)} / {len(r2)} pairs; consistent in both orders {stats}")
    for q, v in res["questions"].items():
        print(f"  {q:<9} rho={v['spearman_rho']} CI={v['rho_ci']} majority picks more novel="
              f"{v['majority_picks_more_novel']} (n={v['n_majority_pairs']})")
    print(f"  vs human majority: {agree}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export").set_defaults(f=export)
    i = sub.add_parser("import")
    i.add_argument("--judge", required=True)
    i.add_argument("--order1", required=True)
    i.add_argument("--order2", required=True)
    i.set_defaults(f=imp)
    a = ap.parse_args()
    a.f(a)
