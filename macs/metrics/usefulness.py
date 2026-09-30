"""Usefulness: pass@k on held-out tests with the unbiased estimator of Chen et al. (2021).

Arush - this is the correctness side. pass@k is the standard unbiased estimate of 'if you drew k
solutions, would at least one be fully correct' on the HELD-OUT tests. correct_and_novel_rate is
the headline number that keeps you honest: a solution only counts if it is BOTH correct AND
novel - an original wrong answer scores nothing.

pass@k = 1 - C(n - c, k) / C(n, k)

where n samples were drawn and c of them pass every held-out test. Computed per
task and averaged. Runtime and memory are logged alongside but are not part of
the measure.
"""
from __future__ import annotations

from math import comb

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator. Returns 1.0 when c > n - k (every k-subset contains a pass)."""
    if n <= 0 or k <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k_over_tasks(n_per_task: list[int], c_per_task: list[int], k: int) -> float:
    return float(np.mean([pass_at_k(n, c, k) for n, c in zip(n_per_task, c_per_task)]))


def correct_and_novel_rate(is_correct: np.ndarray, is_novel: np.ndarray) -> float:
    """Joint rate: fraction of solutions that pass held-out tests AND exceed the novelty percentile."""
    is_correct = np.asarray(is_correct, dtype=bool)
    is_novel = np.asarray(is_novel, dtype=bool)
    if len(is_correct) == 0:
        return 0.0
    return float((is_correct & is_novel).mean())
