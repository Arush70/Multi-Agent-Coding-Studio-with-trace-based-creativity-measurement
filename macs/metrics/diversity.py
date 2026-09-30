"""Diversity of a set of proposals: solution families and set-level spread.

Arush - novelty scores one solution; this scores a whole SET: how many genuinely different
approaches showed up. It clusters solutions into 'families' and reports it four ways (families in
the first 10, a rarefaction curve, entropy, and mean pairwise distance) because a single number
would mislead. The clustering is checked against your hand-written blind labels with the ARI
(that's the SC3 validation, target >= 0.60).

Reported four ways because family count grows with sample size:
  (i)   families among the first n proposals,
  (ii)  a rarefaction curve (families found vs. proposals sampled, averaged over subsamples),
  (iii) Shannon entropy over family assignment,
  (iv)  mean pairwise distance, which needs no clustering at all.

The clustering threshold is calibrated on pilot tasks, frozen, and recorded in
docs/PREREGISTRATION.md. Family detection is validated against blind hand
labels with the adjusted Rand index (SC3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances


def cluster_families(X: np.ndarray, distance_threshold: float, linkage: str = "average") -> np.ndarray:
    """Return an integer family label per row. One row -> one family."""
    if len(X) == 1:
        return np.zeros(1, dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage=linkage
    )
    return model.fit_predict(X)


def n_families(labels: np.ndarray) -> int:
    return int(len(np.unique(labels)))


def family_entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mean_pairwise_distance(X: np.ndarray) -> float:
    if len(X) < 2:
        return 0.0
    D = cosine_distances(X)
    iu = np.triu_indices(len(X), k=1)
    return float(D[iu].mean())


def rarefaction(labels: np.ndarray, rng: np.random.Generator, n_draws: int = 200) -> np.ndarray:
    """Expected number of families found in the first m proposals, m = 1..N, over random orderings."""
    N = len(labels)
    curve = np.zeros(N)
    for _ in range(n_draws):
        perm = rng.permutation(N)
        seen: set[int] = set()
        for m, idx in enumerate(perm):
            seen.add(int(labels[idx]))
            curve[m] += len(seen)
    return curve / n_draws


@dataclass
class DiversityReport:
    n_proposals: int
    families_first_n: int
    n_families_total: int
    entropy: float
    mean_pairwise: float
    rarefaction: list[float]


def diversity_report(X: np.ndarray, distance_threshold: float, *, first_n: int = 10, seed: int = 0) -> DiversityReport:
    labels = cluster_families(X, distance_threshold)
    rng = np.random.default_rng(seed)
    curve = rarefaction(labels, rng)
    return DiversityReport(
        n_proposals=len(X),
        families_first_n=int(round(curve[min(first_n, len(X)) - 1])),
        n_families_total=n_families(labels),
        entropy=family_entropy(labels),
        mean_pairwise=mean_pairwise_distance(X),
        rarefaction=curve.tolist(),
    )


def validate_against_labels(pred_labels: np.ndarray, hand_labels: list[str]) -> float:
    """Adjusted Rand index between automatic families and blind hand labels (SC3 target >= 0.60)."""
    return float(adjusted_rand_score(hand_labels, pred_labels))
