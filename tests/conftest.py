"""Shared test setup.

The Facilitator measures in UniXcoder space, which needs torch and a model download. Tests run
without either: a deterministic stand-in maps each normalised source to a fixed unit vector
(identical code gives identical vectors, different code nearly orthogonal ones), which is what
the convergence logic needs. The real embedding is exercised by scripts/tau_diagnostics.py.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest


def _fake_embed(sources, **_kw):
    rows = []
    for s in sources:
        seed = int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(64)
        rows.append(v / np.linalg.norm(v))
    return np.vstack(rows) if rows else np.zeros((0, 64))


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch):
    from macs import orchestrator
    from macs.metrics import represent
    monkeypatch.setattr(represent, "embed", _fake_embed)
    orchestrator._REFERENCE_CACHE.clear()
    yield
    orchestrator._REFERENCE_CACHE.clear()
