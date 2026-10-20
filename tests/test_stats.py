"""S8 statistics: the REML random-intercept model, planned contrasts, Holm and the MDE."""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from macs.analysis.stats import (
    fit_random_intercept, factorial_fit, facilitator_fit, holm, mde, planned_contrasts)

CODES = {"A": (0, 0), "B": (1, 0), "C": (0, 1), "D": (1, 1)}


def _simulate(seed=0, n_tasks=12, reps=4, effects=(0.5, 0.2, 0.3), task_sd=1.5, noise_sd=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        u = rng.normal(0, task_sd)
        for c, (s, l) in CODES.items():
            for _ in range(reps):
                y = effects[0] * s + effects[1] * l + effects[2] * s * l + u + rng.normal(0, noise_sd)
                rows.append({"task_id": f"t{t}", "condition": c, "y": y})
        for _ in range(reps):
            rows.append({"task_id": f"t{t}", "condition": "D_minus",
                         "y": effects[0] + effects[1] + effects[2] + u - 0.4 + rng.normal(0, noise_sd)})
    return rows


def test_reml_recovers_effects_and_variances_on_average():
    est = []
    for seed in range(60):
        f = factorial_fit(_simulate(seed), "y")
        est.append([*f.beta[1:], f.sigma2, f.sigma2_task])
    m = np.mean(est, axis=0)
    assert np.allclose(m[:3], [0.5, 0.2, 0.3], atol=0.08)
    assert abs(m[3] - 1.0) < 0.1 and abs(m[4] - 2.25) < 0.6


def test_no_task_variance_reduces_to_ordinary_least_squares():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(80), rng.integers(0, 2, 80), rng.integers(0, 2, 80)])
    y = X @ np.array([1.0, 0.5, -0.2]) + rng.normal(size=80)
    groups = np.repeat(np.arange(8), 10)
    f = fit_random_intercept(y, X, groups)
    if f.sigma2_task == 0.0:
        assert np.allclose(f.beta, np.linalg.lstsq(X, y, rcond=None)[0])
    assert f.sigma2_task < 0.2


def test_planned_contrasts_and_confidence_interval_coverage():
    hits = 0
    for seed in range(100):
        cs = {c.label: c for c in planned_contrasts(_simulate(seed, n_tasks=8, reps=3), "y")}
        assert set(cs) == {"y: B - A", "y: D - C", "y: D - D_minus", "y: interaction (D - C) - (B - A)"}
        c = cs["y: D - C"]                                  # truth 0.5 + 0.3 = 0.8
        hits += c.ci_low <= 0.8 <= c.ci_high
    assert 88 <= hits <= 100                                 # nominal 95%


def test_facilitator_contrast_sign():
    c = facilitator_fit(_simulate(1, n_tasks=20, reps=6), "y").contrast({"facilitator": 1})
    assert c.ci_low < 0.4 < c.ci_high and c.estimate > 0      # D - D_minus = +0.4


def test_holm_matches_hand_computation():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    out = holm([0.2, float("nan"), 0.01])
    assert out[0] == pytest.approx(0.2) and np.isnan(out[1]) and out[2] == pytest.approx(0.02)


def test_mde_scales_with_design():
    a = mde(1.0, 24, 6)
    assert a == pytest.approx((1.959964 + 0.841621) * np.sqrt(2 / 144), rel=1e-4)
    assert mde(1.0, 24, 6, n_tests=5) > a > mde(1.0, 48, 6)


def test_not_estimable_without_all_conditions():
    rows = [r for r in _simulate(0) if r["condition"] != "C"]
    assert factorial_fit(rows, "y") is None


def test_stats_report_runs_on_a_summary(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import stats_report
    sims = _simulate(2, n_tasks=4, reps=3)
    sessions = [{"task_id": r["task_id"], "condition": r["condition"], "n_proposals": 5,
                 "n_correct": 4, "correct_and_novel_rate": min(max(r["y"] / 5, 0), 1),
                 "mean_novelty": r["y"], "tokens": 1000 + 100 * r["y"]} for r in sims]
    per_task = {f"t{t}": {"families_at_m": {c: 1.0 + (c in "BD") + 0.1 * t for c in
                                            ("A", "B", "C", "D", "D_minus")}} for t in range(4)}
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"sessions": sessions, "pooled_diversity": {"per_task": per_task}}),
                       encoding="utf-8")
    out = tmp_path / "stats.json"
    monkeypatch.setattr(sys, "argv", ["stats_report", "--summary", str(summary), "--out", str(out)])
    stats_report.main()
    res = json.loads(out.read_text(encoding="utf-8"))
    assert len(res["primary"]) == 5 and all("p_holm" in c for c in res["primary"])


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels not installed")
def test_matches_statsmodels_mixedlm():
    import pandas as pd
    import statsmodels.formula.api as smf
    rows = [r for r in _simulate(5) if r["condition"] in CODES]
    df = pd.DataFrame([{"y": r["y"], "task": r["task_id"], "s": CODES[r["condition"]][0],
                        "l": CODES[r["condition"]][1]} for r in rows])
    ref = smf.mixedlm("y ~ s * l", df, groups=df["task"]).fit(reml=True)
    ours = factorial_fit(rows, "y")
    assert np.allclose(ours.beta, ref.fe_params.values, atol=1e-4)
    assert np.allclose(np.sqrt(np.diag(ours.cov)), ref.bse_fe.values, rtol=1e-3)
    assert ours.sigma2 == pytest.approx(ref.scale, rel=1e-3)


def _simulate_extended(seed=0, n_tasks=16, reps=4, noise_sd=1.0):
    """Core cells plus E, E_minus, S and P with known differences from D."""
    rng = np.random.default_rng(seed)
    truth = {"A": 0.0, "B": 1.0, "C": 0.0, "D": 1.2, "D_minus": 1.0, "E": 0.7, "E_minus": 0.2,
             "S": 0.6, "P": 0.9}
    rows = []
    for t in range(n_tasks):
        u = rng.normal(0, 1.0)
        for c, mu in truth.items():
            for _ in range(reps):
                rows.append({"task_id": f"t{t}", "condition": c, "y": mu + u + rng.normal(0, noise_sd)})
    return rows, truth


def test_extended_contrasts_recover_their_true_values():
    from macs.analysis.stats import PAIRWISE
    est = {}
    for seed in range(40):
        for c in planned_contrasts(_simulate_extended(seed)[0], "y"):
            est.setdefault(c.label, []).append(c.estimate)
    _, truth = _simulate_extended()
    for label, (a, b) in PAIRWISE.items():
        assert abs(np.mean(est[f"y: {label}"]) - (truth[a] - truth[b])) < 0.08, label
    inter = (truth["D"] - truth["D_minus"]) - (truth["E"] - truth["E_minus"])     # 0.2 - 0.5 = -0.3
    assert abs(np.mean(est["y: interaction (D - D_minus) - (E - E_minus)"]) - inter) < 0.1


def test_contrasts_for_conditions_not_run_are_absent():
    rows = [r for r in _simulate_extended(1)[0] if r["condition"] in ("A", "B", "C", "D", "D_minus")]
    labels = {c.label for c in planned_contrasts(rows, "y")}
    assert "y: E - E_minus" not in labels and "y: D - S" not in labels and "y: D - D_minus" in labels


def test_meta_analysis_detects_consistent_and_inconsistent_effects():
    from macs.analysis.stats import meta_analysis
    same = meta_analysis([0.50, 0.55, 0.45], [0.1, 0.1, 0.1])
    assert abs(same["fixed"] - 0.5) < 1e-9 and same["I2"] == 0.0 and same["same_sign"] == 3
    assert same["Q_p"] > 0.5 and same["random_ci"][0] > 0
    diff = meta_analysis([1.0, -1.0, 0.9], [0.1, 0.1, 0.1])
    assert diff["Q_p"] < 0.001 and diff["I2"] > 0.9 and diff["tau2"] > 0
    assert diff["random_ci"][1] - diff["random_ci"][0] > same["random_ci"][1] - same["random_ci"][0]
    assert meta_analysis([0.3], [0.1]) == {"k": 1}
