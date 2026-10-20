"""Rating-study tooling (SC4)."""
import csv
import json

import numpy as np
import pytest

from macs.rating import analysis, forms
from macs.rating.pairs import Solution, build_pairs
from macs.rating.judge import judge_pairs


def _sols(n_tasks=4, per=6):
    out = []
    for t in range(n_tasks):
        for i in range(per):
            out.append(Solution(sid=f"t{t}|s|p{i}", task_id=f"t{t}", session="s", proposal_id=f"p{i}",
                                condition="D" if i % 2 else "A", code=f"x={i}", display=f"x = {i}", novelty=float(i)))
    return out


def test_pairs_are_balanced_and_respect_limits():
    pairs = build_pairs(_sols(), n_pairs=20, seed=1)
    assert len(pairs) == 20
    per_task = {}
    uses = {}
    for p in pairs:
        assert p["A"]["task_id"] == p["B"]["task_id"] and p["A"]["sid"] != p["B"]["sid"]
        per_task[p["task_id"]] = per_task.get(p["task_id"], 0) + 1
        for s in "AB":
            uses[p[s]["sid"]] = uses.get(p[s]["sid"], 0) + 1
    assert max(per_task.values()) - min(per_task.values()) <= 1
    assert max(uses.values()) <= 2
    gaps = sorted(abs(p["A"]["novelty"] - p["B"]["novelty"]) for p in pairs)
    assert gaps[0] <= 1 and gaps[-1] >= 4                        # small and large gaps both present
    assert build_pairs(_sols(), n_pairs=20, seed=1) == pairs      # seeded


def test_bradley_terry_recovers_order():
    rng = np.random.default_rng(0)
    true = {f"s{i}": i * 0.6 for i in range(6)}
    comps = []
    for _ in range(400):
        a, b = rng.choice(list(true), 2, replace=False)
        pa = 1 / (1 + np.exp(true[b] - true[a]))
        comps.append((a, b) if rng.random() < pa else (b, a))
    bt = analysis.bradley_terry(comps)
    assert [k for k, _ in sorted(bt.items(), key=lambda kv: kv[1])] == list(true)


def test_kappas():
    assert analysis.cohen_kappa(list("AABB"), list("AABB")) == 1.0
    assert abs(analysis.cohen_kappa(list("AABB"), list("ABAB"))) < 1e-9
    assert analysis.fleiss_kappa([[3, 0], [0, 3], [3, 0]]) == 1.0


def _pairs():
    out = []
    for i in range(12):
        a = {"sid": f"s{i}a", "novelty": float(i % 4), "display": "a"}
        b = {"sid": f"s{i}b", "novelty": float((i % 4) + 2), "display": "b"}
        out.append({"pair_id": f"P{i + 1:02d}", "task_id": "t", "A": a, "B": b})
    return out


def test_analysis_end_to_end_with_agreeing_raters():
    pairs = _pairs()
    resp = [{"rater": r, "pair_id": p["pair_id"], "question": q, "choice": "B"}
            for r in ("R01", "R02", "R03") for p in pairs for q in ("original", "creative")]
    res = analysis.analyse(pairs, resp, n_boot=50)
    q = res["questions"]["original"]
    assert q["majority_picks_more_novel"] == 1.0 and q["raw_agreement"] == 1.0
    assert q["spearman_rho"] > 0.5


def test_form_export_and_response_import(tmp_path):
    pairs = _pairs()[:3]
    docx_path = forms.export_form(pairs, tmp_path / "f.docx")
    from docx import Document
    text = "\n".join(p.text for p in Document(str(docx_path)).paragraphs)
    assert "Pair 01: Which of these two correct solutions" in text and "Pair 03: Which is the more creative" in text
    html_path = forms.export_booklet(pairs, {"t": "Write f."}, tmp_path / "b.html")
    assert "Pair 02" in html_path.read_text()
    # a Forms-style export
    head = ["ID", forms.CODE_Q, forms.EXPERIENCE_Q] + [forms.question_text(p["pair_id"], k)
                                                       for p in pairs for k in ("original", "creative")]
    with open(tmp_path / "r.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(head)
        w.writerow(["1", "R01", "3 to 5"] + ["A", "B"] * 3)
        w.writerow(["2", "R02", "1 to 2"] + ["B", "", "A", "B", "A", "A"])
    long, exp = forms.read_responses(tmp_path / "r.csv")
    assert exp == {"R01": "3 to 5", "R02": "1 to 2"}
    assert len(long) == 11                                   # one blank answer skipped
    assert {"rater": "R02", "pair_id": "P01", "question": "original", "choice": "B"} in long


class _Judge:
    """Prefers the longer code; first-position bias on ties."""
    def chat(self, messages, **kw):
        u = messages[1]["content"]
        a = u.split("Solution A:")[1].split("Solution B:")[0]
        b = u.split("Solution B:")[1].split("Question:")[0]
        from types import SimpleNamespace
        return SimpleNamespace(text="A" if len(a) >= len(b) else "B")


def test_judge_asks_both_orders_and_drops_position_bias():
    pairs = _pairs()[:2]
    pairs[0]["A"]["display"], pairs[0]["B"]["display"] = "long code here", "short"
    pairs[1]["A"]["display"], pairs[1]["B"]["display"] = "same", "same"       # tie -> position bias
    resp, st = judge_pairs(pairs, {"t": "task"}, _Judge(), log=None)
    assert st["inconsistent"] == 2 and st["consistent"] == 2
    assert {r["pair_id"] for r in resp} == {"P01"} and all(r["choice"] == "A" for r in resp)
