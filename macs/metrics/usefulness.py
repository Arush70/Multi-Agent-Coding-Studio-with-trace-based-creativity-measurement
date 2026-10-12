"""Usefulness: pass@k on held-out tests with the unbiased estimator of Chen et al. (2021).

pass@k estimates the probability that at least one of k drawn solutions passes every held-out
test:

pass@k = 1 - C(n - c, k) / C(n, k)

where n samples were drawn and c of them pass every held-out test. Computed per task and
averaged. Runtime and memory are logged alongside but are not part of the measure.

correct_and_novel_rate is the joint creativity outcome: a solution counts only if it is both
correct and novel, so an original but wrong answer scores nothing.
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
