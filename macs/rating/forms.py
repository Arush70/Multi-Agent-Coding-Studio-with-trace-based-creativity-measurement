"""Rater materials out, responses in.

export_booklet  a self-contained HTML booklet (print to PDF from the browser): the 60 pairs, each
                with the task and the two solutions side by side, labelled Pair 01..60. Raters
                read the booklet and answer in the Microsoft Form.
export_form     a Word document laid out for Microsoft Forms "Quick import": the consent statements
                of the approved consent form (one required question each), the rater code and
                experience questions, the two A/B questions for every pair, and Part B.
read_responses  the Form's Excel (or CSV) export -> one row per (rater, pair, question, choice).
"""
from __future__ import annotations

import csv
import html
import re
from pathlib import Path

Q_ORIGINAL = "Which of these two correct solutions takes the more original approach?"
Q_CREATIVE = "Which is the more creative solution?"
QUESTIONS = {"original": Q_ORIGINAL, "creative": Q_CREATIVE}
EXPERIENCE_Q = "How many years have you been programming in Python?"
EXPERIENCE_OPTIONS = ["less than 1", "1 to 3", "3 to 5", "more than 5"]   # as in the rater guide submitted for ethical review (v1.0)
CODE_Q = "Your rater code (sent to you by the researcher; please do not enter your name)"


INFO = ("This short study is part of an MSc project on how original AI-written code is (University of Exeter "
        "ethics reference 14275550). You will see 60 pairs of short Python solutions and, for each pair, say which "
        "one takes the more original approach and which is the more creative solution. Keep the rating booklet open "
        "alongside this form. It takes about one hour. Please read the Participant Information Sheet (version 1.0, "
        "18 October 2026) before you start. Taking part is voluntary and you can stop at any time; if you close the "
        "form before submitting, nothing is saved. Your ratings are recorded under the rater code the researcher "
        "gives you, never your name.")
CONSENT_Q = "I have read the information above, I am 18 or over, and I agree to take part."
# The consent statements of the approved consent form (version 1.1), one required question each, so
# the form cannot be submitted unless every statement is agreed to.
CONSENT_STATEMENTS = [   # the consent form submitted for ethical review (version 1.0, 18/10/2026)
    "I confirm that I have read the information sheet dated 18/10/2026 (version no. 1.0) for the above "
    "project. I have had the opportunity to consider the information, ask questions and have had these "
    "answered satisfactorily.",
    "I understand that my participation is voluntary and that I am free to withdraw at any time without "
    "giving any reason and without my legal rights being affected. I understand that if I close the form "
    "before submitting, nothing is saved.",
    "I understand that I can ask for my ratings to be deleted by emailing the researcher within two weeks of "
    "taking part, quoting my rater code, and that after this they can no longer be identified or removed.",
    "I understand that relevant sections of the data collected during the study may be looked at by members "
    "of the research team (the researcher and the supervisor) and by individuals from the University of "
    "Exeter, where it is relevant to my taking part in this research.",
    "I understand that taking part involves my rating responses and my years of Python experience, recorded "
    "under my rater code only and anonymised after the withdrawal period, to be used for the purposes of: the "
    "researcher's MSc dissertation; reports published in academic publications and conference presentations; "
    "sharing as a supplementary dataset with the project's results; and storage by the University until "
    "31/12/2031.",
    "I confirm that I am aged 18 or over.",
    "I agree to take part in the above project.",
]


def _short(task_prompt: str, n: int = 900) -> str:
    return task_prompt if len(task_prompt) <= n else task_prompt[:n].rstrip() + " ..."


def export_booklet(pairs: list[dict], prompts: dict[str, str], path: Path, items: list[dict] | None = None) -> Path:
    e = html.escape
    parts = ["""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Rating booklet</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;color:#111;margin:18px;font-size:12px}
h1{font-size:20px;margin:0 0 6px} .pair{break-inside:avoid;page-break-inside:avoid;border-top:2px solid #14213d;padding-top:8px;margin-top:18px}
h2{font-size:15px;margin:4px 0} .task{background:#f3f5f9;padding:6px 8px;white-space:pre-wrap;font-family:Consolas,monospace;font-size:11px}
.two{display:flex;gap:10px;margin-top:6px} .two>div{flex:1;min-width:0}
pre{background:#fafafa;border:1px solid #ddd;padding:6px;font-size:11px;white-space:pre-wrap;margin:2px 0}
.lab{font-weight:bold;color:#14213d} .q{margin-top:4px;color:#333}
@media print{.pair{page-break-before:always}}
</style></head><body>
<h1>Rating booklet</h1>
<p>For each pair, read the task and the two solutions. Both solutions are correct. Then answer the two
questions for that pair in the online form: <b>""" + e(Q_ORIGINAL) + "</b> and <b>" + e(Q_CREATIVE) +
             """</b> There are no right or wrong answers. Comments in the code have been removed.</p>"""]
    for p in pairs:
        parts.append(f"""<div class="pair"><h2>Pair {p['pair_id'][1:]}</h2>
<div class="task">{e(_short(prompts.get(p['task_id'], p['task_id'])))}</div>
<div class="two"><div><div class="lab">Solution A</div><pre>{e(p['A']['display'])}</pre></div>
<div><div class="lab">Solution B</div><pre>{e(p['B']['display'])}</pre></div></div>
<div class="q">Answer in the form: Pair {p['pair_id'][1:]} - original? (A or B) &middot; creative? (A or B)</div></div>""")
    if items:
        parts.append("""<div class="pair"><h1>Part B: which approach?</h1><p>Each item shows one solution. In the form,
choose the approach it uses from the list for that item, or "Something else" if none fits. There are no
right or wrong answers.</p></div>""")
        for it in items:
            opts = "".join(f"<li>{e(o)}</li>" for o in it["options"])
            parts.append(f"""<div class="pair"><h2>Item {it['item_id']}</h2>
<div class="task">{e(_short(prompts.get(it['task_id'], it['task_id']), 500))}</div>
<pre>{e(it['display'])}</pre><div class="q">Answer in the form, choosing one of:</div><ul>{opts}</ul></div>""")
    parts.append("</body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def question_text(pair_id: str, kind: str) -> str:
    return f"Pair {pair_id[1:]}: {QUESTIONS[kind]}"


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PAIR_OPTIONS = ("Solution A", "Solution B")


def export_form(pairs: list[dict], path: Path, items: list[dict] | None = None) -> Path:
    """Word document for Microsoft Forms Quick import, in the layout Quick import recognises as
    choice questions: "Q1. question" on one line, each option on its own line labelled "A.", "B.",
    and so on, and an empty line between questions. Questions without options (the rater code)
    become text questions. No headings or tables, which can merge questions."""
    from docx import Document
    doc = Document()
    doc.add_paragraph("Code originality rating")
    doc.add_paragraph(INFO)
    doc.add_paragraph("Please use the rating booklet to see each pair and item.")
    doc.add_paragraph("")
    n = 1

    def q(text, options=()):
        nonlocal n
        doc.add_paragraph(f"Q{n}. {text}"); n += 1
        for letter, opt in zip(LETTERS, options):
            doc.add_paragraph(f"{letter}. {opt}")
        doc.add_paragraph("")

    for statement in CONSENT_STATEMENTS:          # make each of these required in Forms
        q(statement, ["I agree"])
    q(CODE_Q)
    q(EXPERIENCE_Q, EXPERIENCE_OPTIONS)
    for p in pairs:
        for kind in ("original", "creative"):
            q(question_text(p["pair_id"], kind), PAIR_OPTIONS)
    from macs.rating.labelling import question_text as label_q
    for it in items or []:
        q(label_q(it["item_id"]), it["options"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path


_CHOICE = re.compile(r"^(?:[A-Z][.)]\s*)?(?:solution\s+)?([AB])\b", re.I)


def pair_choice(value: str) -> str:
    """'A', 'B', 'Solution A', 'A. Solution A' -> 'A' / 'B'; anything else -> ''."""
    v = str(value).strip()
    m = re.search(r"solution\s*([AB])\b", v, re.I) or _CHOICE.match(v)
    return m.group(1).upper() if m else ""


_PAIR_COL = re.compile(r"Pair\s*(\d+)", re.I)


def _rows(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook
        ws = load_workbook(path, read_only=True, data_only=True).active
        it = ws.iter_rows(values_only=True)
        head = [str(h or "") for h in next(it)]
        return [dict(zip(head, ["" if v is None else str(v) for v in row])) for row in it]
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_responses(path: Path) -> tuple[list[dict], dict[str, str]]:
    """Responses in long form: [{rater, pair_id, question, choice}], plus {rater: experience}.
    Column headers are matched by their text ('Pair 07' and 'original' / 'creative'), so the
    order of columns in the export does not matter. Blank answers are skipped."""
    long, experience = [], {}
    for i, row in enumerate(_rows(path), start=1):
        rater = next((v.strip() for k, v in row.items() if "rater code" in k.lower() and v), "") or f"rater{i}"
        for k, v in row.items():
            kl = k.lower()
            if ("years of python" in kl or ("years" in kl and "programming in python" in kl)) and v:
                experience[rater] = v.strip()
            m = _PAIR_COL.search(k)
            if not m or not str(v).strip():
                continue
            kind = "original" if "original" in k.lower() else "creative" if "creative" in k.lower() else None
            choice = pair_choice(v)
            if kind and choice in ("A", "B"):
                long.append({"rater": rater, "pair_id": f"P{int(m.group(1)):02d}", "question": kind, "choice": choice})
    return long, experience
