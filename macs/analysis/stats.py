"""S8: mixed-effects models, planned contrasts, Holm adjustment and minimum detectable effect.

The proposal (3.8) analyses each outcome with a linear mixed-effects model: tutor-versus-studio
configuration and number of simulated learners (one or six; two in the proposal, amended
2026-10-23) as fixed effects with their
interaction, and task as a random intercept. Planned contrasts:

  * B - A        studio vs tutor with one learner          (= the studio coefficient)
  * D - C        studio vs tutor with six learners         (= studio + interaction)
  * D - D_minus  Facilitator on vs off (separate model on D and D_minus)
Extensions (pre-registered 2026-10-20): E - E_minus (Facilitator where ideas can converge),
D_minus - E_minus (diversity scaffold), the studio 2 x 2 interaction, and the baselines D - S
(equal token budget) and D - P (single diversity prompt). See PAIRWISE and STUDIO_CODES.

Multiplicity across the planned contrasts on the primary outcomes is controlled with Holm's
step-down procedure. The minimum detectable effect (MDE) for a within-task contrast is derived
from the residual variance estimated on the pilot.

The model y = X b + u_task + e, u ~ N(0, s2_task), e ~ N(0, s2) is fitted by restricted maximum
likelihood (REML). With one random intercept the covariance of task g is
s2 * (I + lam * J), so the REML likelihood can be profiled over the single ratio
lam = s2_task / s2 and maximised numerically. Inference on fixed effects uses Wald t-statistics;
the degrees of freedom are N - G - p + 1 for coefficients that vary within task (all contrasts
here), the usual "containment" choice. Only numpy and scipy are needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

CONDITION_CODES = {          # (studio, six_learners)
    "A": (0, 0), "B": (1, 0), "C": (0, 1), "D": (1, 1),
}


@dataclass
class MixedFit:
    names: list[str]
    beta: np.ndarray
    cov: np.ndarray
    sigma2: float                 # residual variance
    sigma2_task: float            # between-task variance
    n_obs: int
    n_groups: int
    df: int
    reml_loglik: float
    groups_used: list[str] = field(default_factory=list)

    def contrast(self, weights: dict[str, float], label: str = "") -> "Contrast":
        c = np.array([weights.get(n, 0.0) for n in self.names], dtype=float)
        est = float(c @ self.beta)
        se = float(np.sqrt(max(c @ self.cov @ c, 0.0)))
        if se == 0.0:
            t, p = float("nan"), float("nan")
            lo = hi = est
        else:
            t = est / se
            p = float(2 * stats.t.sf(abs(t), self.df))
            q = float(stats.t.ppf(0.975, self.df))
            lo, hi = est - q * se, est + q * se
        return Contrast(label=label, estimate=est, se=se, ci_low=lo, ci_high=hi, t=t, df=self.df, p=p)


@dataclass
class Contrast:
    label: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    t: float
    df: int
    p: float
    p_holm: float | None = None

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _group_index(groups) -> tuple[np.ndarray, list[str]]:
    labels = sorted(set(groups))
    idx = {g: i for i, g in enumerate(labels)}
    return np.array([idx[g] for g in groups]), labels


def fit_random_intercept(y, X, groups, names: list[str] | None = None) -> MixedFit:
    """REML fit of y = X b + u_group + e with one random intercept per group."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    g, labels = _group_index(list(groups))
    n, p = X.shape
    G = len(labels)
    if n <= p:
        raise ValueError("not enough observations for the fixed effects")
    names = names or [f"x{i}" for i in range(p)]
    members = [np.flatnonzero(g == k) for k in range(G)]
    sizes = np.array([len(m) for m in members], dtype=float)

    def pieces(lam: float):
        # H_g^{-1} = I - w_g J with w_g = lam / (1 + n_g lam)
        XtHX = np.zeros((p, p))
        XtHy = np.zeros(p)
        for m, n_g in zip(members, sizes):
            w = lam / (1.0 + n_g * lam)
            Xg, yg = X[m], y[m]
            sx, sy = Xg.sum(axis=0), yg.sum()
            XtHX += Xg.T @ Xg - w * np.outer(sx, sx)
            XtHy += Xg.T @ yg - w * sx * sy
        return XtHX, XtHy

    def neg_reml(log_lam: float) -> float:
        lam = float(np.exp(log_lam))
        XtHX, XtHy = pieces(lam)
        sign, logdet_xhx = np.linalg.slogdet(XtHX)
        if sign <= 0:
            return np.inf
        b = np.linalg.solve(XtHX, XtHy)
        r = y - X @ b
        rHr = 0.0
        for m, n_g in zip(members, sizes):
            w = lam / (1.0 + n_g * lam)
            rg = r[m]
            rHr += rg @ rg - w * rg.sum() ** 2
        s2 = rHr / (n - p)
        if s2 <= 0:
            return np.inf
        logdet_h = float(np.sum(np.log1p(sizes * lam)))
        return 0.5 * ((n - p) * np.log(s2) + logdet_h + logdet_xhx)

    # profile over log(lam); compare with the boundary lam = 0 (no task variance)
    grid = np.linspace(-12, 8, 41)
    vals = [neg_reml(v) for v in grid]
    start = grid[int(np.argmin(vals))]
    res = optimize.minimize_scalar(neg_reml, bounds=(max(start - 2, -14), min(start + 2, 10)), method="bounded")
    log_lam = float(res.x) if res.fun <= min(vals) else float(start)
    lam = float(np.exp(log_lam))
    if lam < 1e-8:
        lam = 0.0
    XtHX, XtHy = pieces(lam)
    beta = np.linalg.solve(XtHX, XtHy)
    r = y - X @ beta
    rHr = sum(r[m] @ r[m] - (lam / (1.0 + n_g * lam)) * r[m].sum() ** 2 for m, n_g in zip(members, sizes))
    s2 = float(rHr / (n - p))
    cov = s2 * np.linalg.inv(XtHX)
    df = max(int(n - G - p + 1), 1)
    return MixedFit(names=names, beta=beta, cov=cov, sigma2=s2, sigma2_task=lam * s2, n_obs=n,
                    n_groups=G, df=df, reml_loglik=float(-neg_reml(np.log(max(lam, 1e-13)))),
                    groups_used=labels)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as given; NaN stays NaN)."""
    p = np.array(pvalues, dtype=float)
    ok = np.flatnonzero(~np.isnan(p))
    adj = np.full_like(p, np.nan)
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


# ------------------------------------------------------------------ the planned analysis

def factorial_fit(rows: list[dict], outcome: str) -> MixedFit | None:
    """Fit outcome ~ studio * six_learners + (1 | task) on conditions A, B, C, D."""
    use = [r for r in rows if r.get("condition") in CONDITION_CODES and r.get(outcome) is not None]
    if len({r["condition"] for r in use}) < 4 or len({r["task_id"] for r in use}) < 2:
        return None
    X = np.array([[1.0, *CONDITION_CODES[r["condition"]],
                   CONDITION_CODES[r["condition"]][0] * CONDITION_CODES[r["condition"]][1]] for r in use])
    y = [float(r[outcome]) for r in use]
    return fit_random_intercept(y, X, [r["task_id"] for r in use],
                                names=["intercept", "studio", "six_learners", "studio:six_learners"])


def facilitator_fit(rows: list[dict], outcome: str) -> MixedFit | None:
    """Fit outcome ~ facilitator + (1 | task) on conditions D and D_minus."""
    use = [r for r in rows if r.get("condition") in ("D", "D_minus") and r.get(outcome) is not None]
    if len({r["condition"] for r in use}) < 2 or len({r["task_id"] for r in use}) < 2:
        return None
    X = np.array([[1.0, 1.0 if r["condition"] == "D" else 0.0] for r in use])
    y = [float(r[outcome]) for r in use]
    return fit_random_intercept(y, X, [r["task_id"] for r in use], names=["intercept", "facilitator"])


def pairwise_fit(rows: list[dict], outcome: str, a: str, b: str) -> MixedFit | None:
    """Fit outcome ~ is_a + (1 | task) on conditions a and b: the difference a - b, with task as
    a random intercept. Used for every two-condition contrast (Facilitator, baselines)."""
    use = [r for r in rows if r.get("condition") in (a, b) and r.get(outcome) is not None]
    if len({r["condition"] for r in use}) < 2 or len({r["task_id"] for r in use}) < 2:
        return None
    X = np.array([[1.0, 1.0 if r["condition"] == a else 0.0] for r in use])
    y = [float(r[outcome]) for r in use]
    return fit_random_intercept(y, X, [r["task_id"] for r in use], names=["intercept", "effect"])


# Studio 2 x 2 (pre-registered 2026-10-20): diversity scaffold (Innovator sees earlier strategies
# and is told to differ) x Facilitator. D = (1, 1), D_minus = (1, 0), E = (0, 1), E_minus = (0, 0).
STUDIO_CODES = {"D": (1, 1), "D_minus": (1, 0), "E": (0, 1), "E_minus": (0, 0)}


def studio_factorial_fit(rows: list[dict], outcome: str) -> MixedFit | None:
    use = [r for r in rows if r.get("condition") in STUDIO_CODES and r.get(outcome) is not None]
    if len({r["condition"] for r in use}) < 4 or len({r["task_id"] for r in use}) < 2:
        return None
    X = np.array([[1.0, *STUDIO_CODES[r["condition"]],
                   STUDIO_CODES[r["condition"]][0] * STUDIO_CODES[r["condition"]][1]] for r in use])
    y = [float(r[outcome]) for r in use]
    return fit_random_intercept(y, X, [r["task_id"] for r in use],
                                names=["intercept", "scaffold", "facilitator", "scaffold:facilitator"])


# Two-condition contrasts, each fitted on its own pair: label -> (a, b)
PAIRWISE = {
    "D - D_minus": ("D", "D_minus"),      # Facilitator, Innovator told to differ (original RQ2 test)
    "E - E_minus": ("E", "E_minus"),      # Facilitator where ideas can converge (RQ2)
    "D_minus - E_minus": ("D_minus", "E_minus"),   # the diversity scaffold alone
    "D - S": ("D", "S"),                  # studio vs tutor sampled to the same token budget
    "D - P": ("D", "P"),                  # studio vs one prompt asking for K different approaches
    "D4 - C4": ("D4", "C4"),              # studio vs tutor with four learners (learner curve)
}


def learner_slope_fit(rows: list[dict], outcome: str, conditions: list[str]) -> MixedFit | None:
    """Fit outcome ~ n_learners + (1 | task) over a learner curve (e.g. A, C4, C = 1, 4, 6
    learners): the change per extra learner. Secondary (amended 2026-10-23)."""
    from macs import conditions as conds
    use = [r for r in rows if r.get("condition") in conditions and r.get(outcome) is not None]
    if len({r["condition"] for r in use}) < 2 or len({r["task_id"] for r in use}) < 2:
        return None
    X = np.array([[1.0, float(conds.get(r["condition"]).n_learners)] for r in use])
    y = [float(r[outcome]) for r in use]
    return fit_random_intercept(y, X, [r["task_id"] for r in use], names=["intercept", "per_learner"])


def planned_contrasts(rows: list[dict], outcome: str) -> list[Contrast]:
    out = []
    f = factorial_fit(rows, outcome)
    if f is not None:
        out.append(f.contrast({"studio": 1}, f"{outcome}: B - A"))
        out.append(f.contrast({"studio": 1, "studio:six_learners": 1}, f"{outcome}: D - C"))
        out.append(f.contrast({"studio:six_learners": 1}, f"{outcome}: interaction (D - C) - (B - A)"))
    for label, (a, b) in PAIRWISE.items():
        g = pairwise_fit(rows, outcome, a, b)
        if g is not None:
            out.append(g.contrast({"effect": 1}, f"{outcome}: {label}"))
    from macs.conditions import LEARNER_CURVE
    for solver, curve in LEARNER_CURVE.items():
        k = learner_slope_fit(rows, outcome, curve)
        if k is not None:
            out.append(k.contrast({"per_learner": 1}, f"{outcome}: per extra learner ({solver}: {', '.join(curve)})"))
    h = studio_factorial_fit(rows, outcome)
    if h is not None:
        out.append(h.contrast({"scaffold:facilitator": 1},
                              f"{outcome}: interaction (D - D_minus) - (E - E_minus)"))
    return out


def mde(sigma2: float, n_tasks: int, reps_per_cell: int, *, alpha: float = 0.05, power: float = 0.80,
        n_tests: int = 1) -> float:
    """Minimum detectable difference between two conditions compared within task.

    The task intercept cancels in a within-task contrast, so its standard error is
    sqrt(2 * sigma2 / (n_tasks * reps_per_cell)). The two-sided level is Holm's strictest,
    alpha / n_tests, so the MDE holds for every contrast in the family.
    """
    se = np.sqrt(2.0 * sigma2 / (n_tasks * reps_per_cell))
    z = stats.norm.ppf(1 - alpha / (2 * n_tests)) + stats.norm.ppf(power)
    return float(z * se)


# ------------------------------------------------------------------ consistency across models

def meta_analysis(estimates: list[float], ses: list[float]) -> dict:
    """Combine one contrast estimated separately on several models.

    Fixed-effect (inverse-variance) and DerSimonian-Laird random-effects pooled estimates, Cochran's
    Q with its p-value, I^2 (share of the spread between models beyond chance), and how many models
    agree in sign with the pooled estimate. Each model is analysed on its own data, so the estimates
    are independent. Needs at least two models with a positive standard error.
    """
    e = np.asarray(estimates, dtype=float)
    s = np.asarray(ses, dtype=float)
    ok = np.isfinite(e) & np.isfinite(s) & (s > 0)
    e, s = e[ok], s[ok]
    k = len(e)
    if k < 2:
        return {"k": int(k)}
    w = 1.0 / s ** 2
    fe = float(np.sum(w * e) / np.sum(w))
    q = float(np.sum(w * (e - fe) ** 2))
    df = k - 1
    p_q = float(stats.chi2.sf(q, df))
    i2 = float(max(0.0, (q - df) / q)) if q > 0 else 0.0
    tau2 = float(max(0.0, (q - df) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))))
    wr = 1.0 / (s ** 2 + tau2)
    re = float(np.sum(wr * e) / np.sum(wr))
    se_re = float(np.sqrt(1.0 / np.sum(wr)))
    z = stats.norm.ppf(0.975)
    return {"k": int(k), "fixed": fe, "random": re, "random_se": se_re,
            "random_ci": [re - z * se_re, re + z * se_re],
            "random_p": float(2 * stats.norm.sf(abs(re / se_re))) if se_re > 0 else float("nan"),
            "Q": q, "Q_p": p_q, "I2": i2, "tau2": tau2,
            "same_sign": int(np.sum(np.sign(e) == np.sign(re))) if re != 0 else 0}
