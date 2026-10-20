"""Rater materials out, responses in.

export_booklet  a self-contained HTML booklet (print to PDF from the browser): the 60 pairs, each
                with the task and the two solutions side by side, labelled Pair 01..60. Raters
                read the booklet and answer in the Microsoft Form.
export_form     a Word document laid out for Microsoft Forms "Quick import": the consent page is
                added separately (the ethics consent form); this document holds the experience
                question and, for every pair, the two A/B questions.
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
EXPERIENCE_Q = "How many years of Python programming experience do you have?"
CODE_Q = "Your rater code (for example R01)"


def _short(task_prompt: str, n: int = 900) -> str:
    return task_prompt if len(task_prompt) <= n else task_prompt[:n].rstrip() + " ..."


def export_booklet(pairs: list[dict], prompts: dict[str, str], path: Path) -> Path:
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
    parts.append("</body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def question_text(pair_id: str, kind: str) -> str:
    return f"Pair {pair_id[1:]}: {QUESTIONS[kind]}"


def export_form(pairs: list[dict], path: Path) -> Path:
    """Word document for Microsoft Forms Quick import: numbered questions, each followed by its
    options on their own lines. Every question is a single choice between A and B."""
    from docx import Document
    doc = Document()
    doc.add_heading("Code originality rating", level=1)
    doc.add_paragraph("Please use the rating booklet to see each pair. Answer A or B for every question.")
    n = 1
    doc.add_paragraph(f"{n}. {CODE_Q}"); n += 1
    doc.add_paragraph(f"{n}. {EXPERIENCE_Q}"); n += 1
    for opt in ("Less than 1", "1 to 2", "3 to 5", "More than 5"):
        doc.add_paragraph(opt)
    for p in pairs:
        for kind in ("original", "creative"):
            doc.add_paragraph(f"{n}. {question_text(p['pair_id'], kind)}"); n += 1
            doc.add_paragraph("A")
            doc.add_paragraph("B")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path


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
            if "years of python" in k.lower() and v:
                experience[rater] = v.strip()
            m = _PAIR_COL.search(k)
            if not m or not str(v).strip():
                continue
            kind = "original" if "original" in k.lower() else "creative" if "creative" in k.lower() else None
            choice = str(v).strip().upper()[:1]
            if kind and choice in ("A", "B"):
                long.append({"rater": rater, "pair_id": f"P{int(m.group(1)):02d}", "question": kind, "choice": choice})
    return long, experience
