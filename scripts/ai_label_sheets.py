"""SC3 labelling sheets for AI raters used through a chat interface (amendment 2026-11-14).

export  writes data/rating/ai_raters/PartB_order1.{html,txt} and PartB_order2.{html,txt}: the same
        48 blind items (task, code, approaches), with the approaches in the listed order (order 1)
        and in reverse (order 2); "Something else" stays last, as in macs/rating/judge.py.
import  reads a rater's two answer files (lines "L01: B") and writes the labels that agree in
        both orders to data/calibration/ai_raters/<rater>.jsonl; disagreements are left out.

Each order is given in a fresh chat with no other context; the exact model name and version, the
date and the full reply are kept with the answers.
"""
from __future__ import annotations

import argparse, html, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from macs.rating import labelling as L, pairs as P  # noqa: E402

OUT = P.RATING_DIR / "ai_raters"
CAL = Path("data/calibration/ai_raters")
LET = "ABCDEFGH"

INSTR = """INSTRUCTIONS FOR THE RATER (please read before answering)

Your role
You are an independent rater in a university research study about program code. Several raters
label the same items separately. Please work only from this document; do not search the web and
do not use anything from earlier conversations.

What you see
There are 48 items, numbered L01 to L48. Each item has:
  - Problem: a short programming problem;
  - Solution: one Python solution to it. Every solution is correct (it passes the problem's tests);
  - Options: a list of approaches, labelled A, B, C, ..., with "Something else / none of these" last.

Your task
For each item, decide which approach the solution uses and choose exactly one option.
  - An approach is the algorithm or main idea used to solve the problem (for example: a loop,
    recursion with a cache, sorting first, a set, a mathematical formula).
  - Judge the method only. Ignore variable names, comments, layout, style, length and efficiency
    details that do not change the method.
  - If the solution combines several ideas, choose the option that describes its main idea, the
    one that does the essential work.
  - Choose "Something else / none of these" only if none of the listed approaches describes the
    main idea.
  - You do not need to run the code. You are not told how or by whom the solutions were written;
    please do not guess.
  - The options are listed separately for each item; their letters differ between items. Use the
    letter shown under that item.

How to answer
Answer every item, from L01 to L48. Reply with exactly 48 lines and nothing else: no
explanations, no headings, no code blocks. Each line has the item number, a colon, a space and one
letter, like this:
L01: B
L02: A
L03: E
...
L48: D"""

END = """END OF ITEMS. Now give your answer: exactly 48 lines, L01 to L48, each in the form
"L01: B", with no other text."""


def ordered(it, order):
    fams = [o for o in it["options"] if o != L.OTHER]
    return (fams if order == 1 else fams[::-1]) + [L.OTHER]


def export(_a):
    items = L.load_items()
    prompts = {t: P.task_prompt(t) for t in {i["task_id"] for i in items}}
    OUT.mkdir(parents=True, exist_ok=True)
    for order in (1, 2):
        txt = [f"SC3 approach labelling - order {order} of 2 - {len(items)} items", "", INSTR, ""]
        hb = []
        for it in items:
            opts = ordered(it, order)
            txt += ["=" * 70, f"Item {it['item_id']}", "", "Problem:", prompts[it["task_id"]].strip(), "",
                    "Solution:", it["display"], "", "Options:"]
            txt += [f"{LET[k]}. {o}" for k, o in enumerate(opts)] + [""]
            e = html.escape
            hb.append(f"<div class='it'><h2>Item {it['item_id']}</h2><div class='task'>{e(prompts[it['task_id']].strip())}</div>"
                      f"<pre>{e(it['display'])}</pre><ol type='A'>" + "".join(f"<li>{e(o)}</li>" for o in opts) + "</ol></div>")
        txt += ["=" * 70, END]
        (OUT / f"PartB_order{order}.txt").write_text("\n".join(txt), encoding="utf-8")
        (OUT / f"PartB_order{order}.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>SC3 approach labelling, order {order}</title><style>
@page{{size:A4;margin:14mm}} body{{font-family:Arial,sans-serif;font-size:10pt;color:#111}}
h1{{font-size:15pt}} h2{{font-size:11.5pt;margin:0 0 4pt}} .ins{{white-space:pre;background:#f2f4f7;padding:8pt;font-family:Consolas,monospace;font-size:8pt;line-height:1.35}}
.it{{page-break-inside:avoid;border-top:1px solid #ccc;padding-top:6pt;margin-top:8pt}}
.task{{white-space:pre-wrap;font-size:9pt;color:#333;margin-bottom:4pt}}
pre{{font-family:Consolas,monospace;font-size:8.3pt;background:#f6f7f9;padding:5pt;white-space:pre-wrap}}
ol{{margin:3pt 0 0 0}}</style></head><body>
<h1>SC3 approach labelling: order {order} of 2 ({len(items)} items)</h1>
<div class="ins">{html.escape(INSTR)}</div>{''.join(hb)}<div class="ins" style="margin-top:10pt">{html.escape(END)}</div></body></html>""", encoding="utf-8")
    print("written to", OUT)


def parse(path):
    got = {}
    for m in re.finditer(r"\bL(\d{2})\s*[:.)-]\s*\(?([A-H])\b", Path(path).read_text(encoding="utf-8")):
        got[f"L{m.group(1)}"] = m.group(2)
    return got


def imp(a):
    items = {i["item_id"]: i for i in L.load_items()}
    r1, r2 = parse(a.order1), parse(a.order2)
    CAL.mkdir(parents=True, exist_ok=True)
    rows, agree = [], 0
    for iid, it in items.items():
        o1, o2 = ordered(it, 1), ordered(it, 2)
        c1 = o1[LET.index(r1[iid])] if iid in r1 and LET.index(r1[iid]) < len(o1) else None
        c2 = o2[LET.index(r2[iid])] if iid in r2 and LET.index(r2[iid]) < len(o2) else None
        ok = c1 is not None and c1 == c2
        agree += ok
        rows.append({"item_id": iid, "task_id": it["task_id"], "session": it["session"],
                     "proposal_id": it["proposal_id"], "rater": a.rater, "order1": c1, "order2": c2,
                     "label": L._family(c1, it) if ok else None})
    out = CAL / f"{a.rater}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{a.rater}: {len(r1)}/{len(items)} answered in order 1, {len(r2)} in order 2; "
          f"{agree} agree in both orders -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export").set_defaults(f=export)
    i = sub.add_parser("import")
    i.add_argument("--rater", required=True, help="short name, e.g. claude_fable, chatgpt, mistral")
    i.add_argument("--order1", required=True)
    i.add_argument("--order2", required=True)
    i.set_defaults(f=imp)
    a = ap.parse_args()
    a.f(a)
