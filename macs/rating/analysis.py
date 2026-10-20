"""Analysis of the rating study (SC4), following the pre-registration.

* Bradley-Terry strengths for every rated solution, per question (Hunter's MM algorithm, with a
  weak prior of one half-win against a reference solution so undefeated items stay finite).
* Agreement between raters: mean pairwise Cohen's kappa and Fleiss' kappa on the A/B choices
  (target 0.60 or more), with raw agreement.
* Validity: Spearman's rho between the Bradley-Terry strength and the automatic novelty z
  (target 0.50 or more), with a bootstrap 95% interval over pairs; and how often the majority
  choice picks the solution with the higher automatic novelty.
The AI judge is analysed the same way but reported separately, never pooled with the humans.
"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict

import numpy as np
from scipy import stats


def bradley_terry(comparisons: list[tuple[str, str]], n_iter: int = 2000, tol: float = 1e-10,
                  prior: float = 0.5) -> dict[str, float]:
    """comparisons: (winner, loser) pairs. Returns log-strength per item (mean 0)."""
    items = sorted({x for c in comparisons for x in c})
    if not items:
        return {}
    idx = {x: i for i, x in enumerate(items)}
    n = len(items)
    wins = np.full(n, prior)                      # prior: half a win ...
    games = defaultdict(float)
    for w, l in comparisons:
        wins[idx[w]] += 1
        games[(idx[w], idx[l])] += 1
        games[(idx[l], idx[w])] += 1
    p = np.ones(n)
    for _ in range(n_iter):
        denom = 1.0 / (p + 1.0)                   # ... from one game against a reference of strength 1
        for (i, j), g in games.items():
            denom[i] += g / (p[i] + p[j])
        new = wins / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - p)) < tol:
            p = new
            break
        p = new
    logp = np.log(p)
    return {x: float(logp[idx[x]]) for x in items}


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if len(a) < 2:
        return None
    po = np.mean([x == y for x, y in zip(a, b)])
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (len(a) ** 2)
    return 1.0 if pe == 1 else float((po - pe) / (1 - pe))


def fleiss_kappa(table: list[list[int]]) -> float | None:
    """table[i] = counts per category for item i (same number of raters per item)."""
    t = np.asarray(table, dtype=float)
    if t.ndim != 2 or len(t) < 2:
        return None
    n = t.sum(axis=1)
    if not np.all(n == n[0]) or n[0] < 2:
        return None
    n = n[0]
    p_j = t.sum(axis=0) / t.sum()
    p_i = (np.sum(t * t, axis=1) - n) / (n * (n - 1))
    pe = np.sum(p_j ** 2)
    return 1.0 if pe == 1 else float((p_i.mean() - pe) / (1 - pe))


def analyse(pairs: list[dict], responses: list[dict], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Full SC4 analysis for one set of raters (humans, or the judge alone)."""
    by_pair = {p["pair_id"]: p for p in pairs}
    out = {"raters": sorted({r["rater"] for r in responses}), "questions": {}}
    for q in ("original", "creative"):
        rs = [r for r in responses if r["question"] == q and r["pair_id"] in by_pair]
        comps = []
        for r in rs:
            p = by_pair[r["pair_id"]]
            w, l = (p["A"]["sid"], p["B"]["sid"]) if r["choice"] == "A" else (p["B"]["sid"], p["A"]["sid"])
            comps.append((w, l))
        bt = bradley_terry(comps)
        nov = {p[s]["sid"]: p[s]["novelty"] for p in pairs for s in ("A", "B")}
        common = [s for s in bt if s in nov]
        rho = stats.spearmanr([bt[s] for s in common], [nov[s] for s in common])[0] if len(common) >= 3 else None
        # bootstrap over pairs
        rng = np.random.default_rng(seed)
        pair_ids = sorted({r["pair_id"] for r in rs})
        boots = []
        if rho is not None and len(pair_ids) >= 5:
            for _ in range(n_boot):
                pick = rng.choice(pair_ids, size=len(pair_ids), replace=True)
                cnt = Counter(pick)
                sub = []
                for r in rs:
                    sub += [r] * cnt.get(r["pair_id"], 0)
                c2 = []
                for r in sub:
                    p = by_pair[r["pair_id"]]
                    c2.append((p["A"]["sid"], p["B"]["sid"]) if r["choice"] == "A" else (p["B"]["sid"], p["A"]["sid"]))
                b2 = bradley_terry(c2, n_iter=300, tol=1e-6)
                cc = [s for s in b2 if s in nov]
                if len(cc) >= 3:
                    v = stats.spearmanr([b2[s] for s in cc], [nov[s] for s in cc])[0]
                    if np.isfinite(v):
                        boots.append(v)
        # agreement
        raters = sorted({r["rater"] for r in rs})
        choice = {(r["rater"], r["pair_id"]): r["choice"] for r in rs}
        kappas = []
        for a, b in itertools.combinations(raters, 2):
            both = [pid for pid in pair_ids if (a, pid) in choice and (b, pid) in choice]
            k = cohen_kappa([choice[(a, pid)] for pid in both], [choice[(b, pid)] for pid in both])
            if k is not None:
                kappas.append(k)
        full = [pid for pid in pair_ids if all((r, pid) in choice for r in raters)]
        table = [[sum(choice[(r, pid)] == c for r in raters) for c in ("A", "B")] for pid in full]
        raw = (np.mean([len({choice[(r, pid)] for r in raters}) == 1 for pid in full]) if full and len(raters) > 1
               else None)
        # does the majority pick the more novel solution?
        hits = []
        for pid in pair_ids:
            votes = Counter(choice[(r, pid)] for r in raters if (r, pid) in choice)
            if not votes or votes["A"] == votes["B"]:
                continue
            maj = "A" if votes["A"] > votes["B"] else "B"
            p = by_pair[pid]
            if p["A"]["novelty"] == p["B"]["novelty"]:
                continue
            more_novel = "A" if p["A"]["novelty"] > p["B"]["novelty"] else "B"
            hits.append(maj == more_novel)
        out["questions"][q] = {
            "n_responses": len(rs), "n_pairs": len(pair_ids), "n_solutions": len(bt),
            "spearman_rho": None if rho is None or not np.isfinite(rho) else float(rho),
            "rho_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
            "mean_cohen_kappa": float(np.mean(kappas)) if kappas else None,
            "fleiss_kappa": fleiss_kappa(table) if len(raters) > 1 else None,
            "raw_agreement": None if raw is None else float(raw),
            "majority_picks_more_novel": float(np.mean(hits)) if hits else None, "n_majority_pairs": len(hits),
            "bradley_terry": bt,
        }
    return out


def judge_agreement(human: list[dict], judge: list[dict]) -> dict:
    """How often the AI judge's choice equals the human majority, per question."""
    out = {}
    for q in ("original", "creative"):
        votes = defaultdict(Counter)
        for r in human:
            if r["question"] == q:
                votes[r["pair_id"]][r["choice"]] += 1
        jc = {r["pair_id"]: r["choice"] for r in judge if r["question"] == q}
        hits = []
        for pid, v in votes.items():
            if pid in jc and v["A"] != v["B"]:
                hits.append(jc[pid] == ("A" if v["A"] > v["B"] else "B"))
        out[q] = {"agreement_with_majority": float(np.mean(hits)) if hits else None, "n": len(hits)}
    return out
