"""Model-relative novelty: k-nearest-neighbour distance to a frozen reference set.

For each solution, the mean cosine distance to its k nearest neighbours in the task's
reference set, z-scored against the reference set's own leave-one-out kNN distances, so
0 = typical and 2 = two standard deviations further out. Nearest-neighbour rather than
centroid distance because the reference is multimodal (Russell et al., 2026, Appendix H
uses the same construction with k = 25 on 300-dimensional feature vectors).

Novel is not the same as good: novelty is always paired with correctness. The reference set
is sampled once per task and frozen; the system's own outputs are never part of it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler

K_DEFAULT = 5
DEGENERATE_STD = 1e-6     # below this the reference LOO kNN distances have no usable spread


@dataclass
class ReferenceStats:
    """Everything needed to score new solutions against a frozen reference."""
    X_ref: np.ndarray            # (n_ref, d) already in the representation space
    scaler: StandardScaler | None
    k: int
    ref_knn_mean: float
    ref_knn_std: float

    @classmethod
    def fit(cls, X_ref: np.ndarray, k: int = K_DEFAULT, standardise: bool = False) -> "ReferenceStats":
        scaler = None
        if standardise:                      # sensible for AST features, not for unit-norm embeddings
            scaler = StandardScaler().fit(X_ref)
            X_ref = scaler.transform(X_ref)
        D = cosine_distances(X_ref, X_ref)
        np.fill_diagonal(D, np.inf)          # leave-one-out
        loo = np.sort(D, axis=1)[:, :k].mean(axis=1)
        return cls(X_ref=X_ref, scaler=scaler, k=k, ref_knn_mean=float(loo.mean()), ref_knn_std=float(loo.std() + 1e-12))

    def raw_knn(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            X = self.scaler.transform(X)
        D = cosine_distances(X, self.X_ref)
        return np.sort(D, axis=1)[:, :self.k].mean(axis=1)

    @property
    def degenerate(self) -> bool:
        """True when the reference's own kNN distances have no spread (e.g. the model wrote the
        same solution almost every time), so a z-score has no scale. Amended 2026-10-25."""
        return self.ref_knn_std < DEGENERATE_STD

    def novelty(self, X: np.ndarray) -> np.ndarray:
        """z-scored kNN distance: 0 = as typical as the reference; 2 = two SDs further out.
        NaN for a degenerate reference (no spread): the percentile outcome still applies."""
        if self.degenerate:
            return np.full(len(X), np.nan)
        return (self.raw_knn(X) - self.ref_knn_mean) / self.ref_knn_std

    def percentile(self, X: np.ndarray) -> np.ndarray:
        """Rarity percentile relative to the reference's own LOO distances (as in StoryScope)."""
        D = cosine_distances(self.X_ref, self.X_ref)
        np.fill_diagonal(D, np.inf)
        loo = np.sort(D, axis=1)[:, :self.k].mean(axis=1)
        raw = self.raw_knn(X)
        return np.array([(loo < r).mean() for r in raw])


def above_percentile(novelty_pct: np.ndarray, threshold: float = 0.90) -> np.ndarray:
    """Boolean mask used for the correct-and-novel joint rate."""
    return novelty_pct >= threshold
