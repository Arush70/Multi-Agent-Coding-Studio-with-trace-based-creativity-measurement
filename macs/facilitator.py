"""Facilitator: convergence detection and intervention.

Arush - the Facilitator is the heart of your hypothesis. It watches the studio's proposals
come in and measures whether they are CONVERGING (each new idea landing close to an earlier
one). When the recent proposals stop being novel - their average nearest-neighbour distance
over a short window drops below a threshold tau - it intervenes, telling the Innovator to try
a genuinely different approach. The four intervention types rotate so you can compare them.

The one rule that keeps this honest: tau is NOT tuned to make results look good. It is fixed
by a pre-registered rule - the 25th percentile of the reference set's own nearest-neighbour
distances - so the Facilitator's firing threshold is decided by the task, not by you.
"""
from __future__ import annotations

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
        if len(self._nn_dists) < self.window:
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

    def intervene(self, *, families_so_far: list[str], last_two: tuple[str, str] | None) -> tuple[str, dict]:
        """Return (intervention_type, format-args for the corresponding prompt in roles.yaml)."""
        kind = self.next_intervention()
        if kind == "constraint":
            args = {"excluded": families_so_far[-1] if families_so_far else "the data structure used so far"}
        elif kind == "exclusion":
            args = {"family": families_so_far[-1] if families_so_far else "the current approach"}
        elif kind == "hybridisation":
            a, b = last_two if last_two else ("the first proposal", "the second proposal")
            args = {"a": a, "b": b}
        else:
            args = {}
        return kind, args
