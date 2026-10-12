"""Facilitator: convergence detection and intervention.

The Facilitator watches the studio's proposals and measures whether they are converging (each
new idea landing close to an earlier one). When the mean nearest-neighbour distance over the
last few proposals drops below a threshold tau, it intervenes and tells the Innovator to try a
genuinely different approach. The four intervention types rotate so they can be compared.

tau is not tuned to the results. It is fixed by a pre-registered rule, the 25th percentile of
the reference set's own nearest-neighbour distances, so the firing threshold is decided by the
task.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from itertools import cycle

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

INTERVENTIONS = ("constraint", "dissent", "exclusion", "hybridisation")


def tau_from_reference(X_ref: np.ndarray, percentile: float = 25.0) -> float:
    """Pre-registered rule for tau: a low percentile of the reference set's own LOO-NN distances."""
    D = cosine_distances(X_ref, X_ref)
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    return float(np.percentile(nn, percentile))


@dataclass
class Facilitator:
    tau: float
    window: int = 3
    enabled: bool = True                      # False in condition D_minus
    _history: list[np.ndarray] = field(default_factory=list)
    _nn_dists: list[float] = field(default_factory=list)
    _rotation: "cycle" = field(default_factory=lambda: cycle(INTERVENTIONS))
    n_interventions: int = 0

    def observe(self, embedding: np.ndarray) -> float | None:
        """Record a new proposal; return its distance to the nearest earlier proposal (None for the first)."""
        emb = np.asarray(embedding).reshape(1, -1)
        d = None
        if self._history:
            H = np.vstack(self._history)
            d = float(cosine_distances(emb, H).min())
            self._nn_dists.append(d)
        self._history.append(emb.ravel())
        return d

    def window_metric(self) -> float | None:
        """Mean nearest-neighbour distance over the last `window` proposals.

        The window is counted in proposals, not distances: once `window` proposals exist, the
        metric is the mean of their distances to earlier proposals (the first proposal has no
        earlier neighbour, so the first window averages window - 1 distances). With window 3
        and K = 5 the Facilitator can fire after proposal 3 and shape proposals 4 and 5.
        """
        if len(self._history) < self.window:
            return None
        return float(np.mean(self._nn_dists[-self.window:]))

    def should_intervene(self) -> tuple[bool, float | None]:
        m = self.window_metric()
        if not self.enabled or m is None:
            return False, m
        return m < self.tau, m

    def next_intervention(self) -> str:
        self.n_interventions += 1
        return next(self._rotation)

    def intervene(self, *, construct: str | None = None, approach: str | None = None,
                  last_two: tuple[str, str] | None = None,
                  families_so_far: list[str] | None = None) -> tuple[str, dict]:
        """Return (intervention_type, format-args for the corresponding prompt in roles.yaml).

        construct: what the converging implementations share (e.g. "a set"), from
                   dominant_construct(); approach: the name of the latest approach, from
                   approach_name(); last_two: names of the two latest approaches (hybridisation).
        families_so_far is the old argument, kept so older callers still work.
        """
        kind = self.next_intervention()
        if families_so_far and not construct:
            construct = families_so_far[-1]
        if families_so_far and not approach:
            approach = families_so_far[-1]
        if kind == "constraint":
            args = {"excluded": construct or "the data structure used so far"}
        elif kind == "exclusion":
            label = approach or "the current approach"
            if construct:
                label += f", built on {construct}"
            args = {"family": label}
        elif kind == "hybridisation":
            a, b = last_two if last_two else ("the first proposal", "the second proposal")
            args = {"a": a, "b": b}
        else:
            args = {}
        return kind, args


# --------------------------------------------------------------------------------------------
# Naming what the proposals have converged on, so interventions are specific.
#
# These helpers read what the converging proposals have in common from the code (which
# construct the recent implementations share) and from the Innovator's strategy text (its
# approach name), so the constraint and exclusion prompts are specific, e.g. "Propose a strategy
# that does NOT use a set", rather than generic.

# (label used in the prompt, detector) in priority order; ties go to the earlier entry
_CONSTRUCTS = ("a set", "a dictionary / hash map", "sorting", "recursion", "a heap",
               "bit manipulation", "nested loops")


def code_constructs(code: str) -> set[str]:
    """Which of the named constructs a piece of code uses. Unparseable code -> empty set."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    found: set[str] = set()

    def call_name(n: ast.Call) -> str:
        f = n.func
        return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""

    for n in ast.walk(tree):
        if isinstance(n, (ast.Set, ast.SetComp)):
            found.add("a set")
        elif isinstance(n, (ast.Dict, ast.DictComp)):
            found.add("a dictionary / hash map")
        elif isinstance(n, ast.Call):
            name = call_name(n)
            if name in ("set", "frozenset"):
                found.add("a set")
            elif name in ("dict", "Counter", "defaultdict", "OrderedDict"):
                found.add("a dictionary / hash map")
            elif name in ("sorted", "sort"):
                found.add("sorting")
            elif name.startswith("heap") or name in ("nlargest", "nsmallest"):
                found.add("a heap")
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
            found.add("bit manipulation")
        elif isinstance(n, (ast.For, ast.While)):
            if any(isinstance(m, (ast.For, ast.While)) for b in n.body for m in ast.walk(b)):
                found.add("nested loops")
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == fn.name
                   for c in ast.walk(fn)):
                found.add("recursion")
    return found


def dominant_construct(codes: list[str]) -> str | None:
    """The construct used by the most of `codes` (ties -> priority order); None if none is used."""
    counts = {c: 0 for c in _CONSTRUCTS}
    for code in codes:
        for c in code_constructs(code):
            counts[c] += 1
    best = max(_CONSTRUCTS, key=lambda c: (counts[c], -_CONSTRUCTS.index(c)))
    return best if counts[best] > 0 else None


def approach_name(strategy: str, max_words: int = 12) -> str:
    """Short name of a strategy: its first sentence, cut to `max_words` words."""
    text = " ".join(strategy.strip().split())
    cuts = [text.find(stop) for stop in (". ", ": ", "; ", " - ") if stop in text]
    if cuts:
        text = text[:min(cuts)]
    text = text.strip("*#` ")          # models often bold the name: **Hash set approach**
    words = text.rstrip(".").split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
