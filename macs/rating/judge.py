"""The independent AI judge (ethics application, Method 4).

A model from a different family to the one that wrote the code (DeepSeek-Coder-V2 Lite, local and
free) answers the raters' two questions for every pair. Each pair is asked in BOTH orders (A/B and
B/A), because models prefer whichever answer comes first; the judge's choice counts only when the
two orders agree, otherwise it is recorded as inconsistent and left out. Reported separately from
the human ratings, never combined with them.
"""
from __future__ import annotations

import re

from macs.rating.forms import QUESTIONS

JUDGE_PROMPT = (
    "You are an experienced Python programmer judging code. Both solutions below are correct. "
    "Answer the question with a single letter, A or B, on the first line, then at most one sentence.")
_ANS = re.compile(r"\b([AB])\b")


def _ask(client, task: str, first: str, second: str, question: str, seed: int) -> str | None:
    user = (f"Task:\n{task}\n\nSolution A:\n```python\n{first}\n```\n\nSolution B:\n```python\n{second}\n```\n\n"
            f"Question: {question}\nReply A or B.")
    r = client.chat([{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
                    seed=seed, temperature=0.0, max_tokens=60)
    m = _ANS.search(r.text.strip().splitlines()[0] if r.text.strip() else "")
    return m.group(1) if m else None


def judge_pairs(pairs: list[dict], prompts: dict[str, str], client, *, log=print) -> tuple[list[dict], dict]:
    """Returns (responses in the same long form as the humans', with rater 'judge'), stats."""
    out, inconsistent, unparsed = [], 0, 0
    for i, p in enumerate(pairs, start=1):
        task = prompts.get(p["task_id"], p["task_id"])
        for kind, question in QUESTIONS.items():
            fwd = _ask(client, task, p["A"]["display"], p["B"]["display"], question, seed=1000 + i)
            rev = _ask(client, task, p["B"]["display"], p["A"]["display"], question, seed=2000 + i)
            if fwd is None or rev is None:
                unparsed += 1
                continue
            rev_as_ab = "A" if rev == "B" else "B"          # in the reversed order, A meant B
            if fwd != rev_as_ab:
                inconsistent += 1
                continue
            out.append({"rater": "judge", "pair_id": p["pair_id"], "question": kind, "choice": fwd})
        if log and i % 10 == 0:
            log(f"[{i}/{len(pairs)}] judged")
    n = 2 * len(pairs)
    return out, {"asked": n, "consistent": len(out), "inconsistent": inconsistent, "unparsed": unparsed,
                 "position_consistency": len(out) / n if n else None}
