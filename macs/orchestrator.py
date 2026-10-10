"""Session orchestrator - the loop that turns model calls into a trace.

Arush - this ties everything together: it takes a task, runs a session in one of the five
conditions, and writes every step to a trace. Two families of condition:

    A  1 learner  + Tutor            C  2 learners + Tutor          (Tutor: one solution)
    B  1 learner  + Studio           D  2 learners + Studio         (Studio: Innovator ->
    D_minus  2 learners + Studio, Facilitator disabled              Engineer -> Verifier loop,
                                                                    Critic + Facilitator)

The Studio is your contribution: the Innovator proposes several strategies, the Engineer
implements each, the Verifier runs the sandbox, and the Facilitator watches for the ideas
converging and pushes for a fresh one. D_minus runs the same studio with the Facilitator
switched OFF - that single difference is how you isolate the Facilitator's effect.

Orchestration is plain Python on purpose: every control-flow decision your results depend on
is visible in this one file. The one rule enforced here: the sandbox is run on
task.public_tests ONLY, so held-out tests can never reach a prompt.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from macs.llm.client import prompt_hash
from macs.sandbox import runner
from macs.tasks.bank import Task
from macs.trace.writer import TraceWriter
from macs.facilitator import Facilitator, tau_from_reference, dominant_construct, approach_name
from macs.metrics.represent import ast_features, normalise
from macs.metrics.diversity import cluster_families

PROMPTS = yaml.safe_load((Path(__file__).parent / "prompts" / "roles.yaml").read_text())
PERSONAS = yaml.safe_load((Path(__file__).parent / "prompts" / "personas.yaml").read_text())["personas"]

# Arush - bump when the session logic changes in a way that makes older traces incomparable.
# It is stamped into every trace header; the analysis skips traces from older versions by
# default (scripts/analyze.py --include-old to keep them). 0.2.0 = fixes of 2026-10-07/09.
ORCHESTRATOR_VERSION = "0.2.0"

K_PROPOSALS = 5
# Critique-revision rounds (proposal: "bounded critique-revision, max 3 rounds"). The studio
# always runs at least one round; it stops early once a revision passes all public tests.
MAX_REVISIONS = 3
# Token ceiling per session, same for every condition (proposal: "all conditions share stopping
# rules and token ceilings"). Counted as prompt + completion tokens over all model calls. When the
# next call would start at or above it, the session stops and session_end says so.
TOKEN_CEILING = 40_000
CONVERGENCE_WINDOW = 3
# Provisional tau, used when a task has no frozen reference set yet (pilot) or its reference is
# degenerate (tau = 0). The trace header records tau and tau_source for every studio session.
# (The family-clustering threshold lives in macs/analysis/aggregate.py - one value, one place.)
DEFAULT_TAU = 0.05

CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def _embed(code: str) -> np.ndarray | None:
    # Represent a solution as AST features (no heavy model needed) so the Facilitator can
    # measure how close proposals are. Code that doesn't parse yet returns None and is skipped.
    try:
        return ast_features(normalise(code))
    except Exception:  # noqa: BLE001
        return None


@dataclass
class SessionConfig:
    condition: str                    # A | B | C | D | D_minus
    seed: int
    sandbox_backend: str = "docker"   # real runs use docker; dev on Windows uses "subprocess"
    n_learners: int = 1
    k_proposals: int = K_PROPOSALS
    tau: float | None = None          # None -> resolved from the reference set, else DEFAULT_TAU
    token_ceiling: int = TOKEN_CEILING
    max_revisions: int = MAX_REVISIONS
    reference_n: int | None = field(default=None, init=False)   # size of the reference used for tau

    def __post_init__(self) -> None:
        self.n_learners = 2 if self.condition in ("C", "D", "D_minus") else 1

    @property
    def is_studio(self) -> bool:
        return self.condition in ("B", "D", "D_minus")

    @property
    def facilitator_enabled(self) -> bool:
        return self.condition in ("B", "D")          # OFF only in D_minus

    def resolve_tau(self, task: Task) -> tuple[float, str]:
        """Return (tau, tau_source). Uses the pre-registered reference rule when the task has a
        frozen reference set; otherwise a provisional default. tau_source is written to the trace
        header so every session records where its threshold came from.

        Arush - two fixes (2026-10-07, docs/DEV_LOG.md):
          * tau is now computed in the SAME space the Facilitator measures in (raw AST features,
            cosine). It used to be read off the reference's STANDARDISED features, a different
            space, so the number was not comparable to the distances it was compared against.
          * if the reference is degenerate (mostly duplicate solutions, so the 25th-percentile
            nearest-neighbour distance is 0), tau = 0 and the Facilitator can never fire. The
            proposal says tau is checked for a non-degenerate firing rate; a zero tau fails that
            check, so we fall back to the documented default and say so in the trace.
        """
        if self.tau is not None:
            return self.tau, "override"
        try:
            from macs.metrics.reference import build_reference_stats
            rs = build_reference_stats(task.id, standardise=False)
        except Exception:  # noqa: BLE001 - no reference set yet (pilot); use the default
            return DEFAULT_TAU, "default:no_reference"
        self.reference_n = int(rs.X_ref.shape[0])
        tau = tau_from_reference(rs.X_ref)
        if tau <= 1e-9:
            return DEFAULT_TAU, "default:degenerate_reference"
        return tau, "reference"


def _header(client, cfg: "SessionConfig", extra: dict | None = None) -> dict:
    return {
        **client.profile.identity,
        "seed": cfg.seed,
        "prompt_hashes": {k: prompt_hash(v) for k, v in PROMPTS.items() if isinstance(v, str)},
        "token_ceiling": cfg.token_ceiling,
        "max_revision_rounds": cfg.max_revisions if cfg.is_studio else 0,
        "orchestrator_version": ORCHESTRATOR_VERSION,
        **(extra or {}),
    }


def _pick_personas(cfg: SessionConfig) -> list[dict]:
    start = cfg.seed % len(PERSONAS)
    return [PERSONAS[(start + j) % len(PERSONAS)] for j in range(cfg.n_learners)]


class TokenBudgetExceeded(RuntimeError):
    """Raised before a model call that would start at or above the session's token ceiling."""


class _BudgetedClient:
    """Wraps the model client and counts tokens; refuses a call once the ceiling is reached.

    Arush - added 2026-10-09 (docs/DEV_LOG.md). The ceiling used to be written into the header
    but never enforced. Cached responses count too (they are the same tokens), so a re-run from
    the cache stops at exactly the same point.
    """
    def __init__(self, client, ceiling: int):
        self._client = client
        self.profile = client.profile
        self.ceiling = ceiling
        self.used = 0

    def chat(self, messages, **kw):
        if self.used >= self.ceiling:
            raise TokenBudgetExceeded(f"token ceiling {self.ceiling} reached ({self.used} used)")
        comp = self._client.chat(messages, **kw)
        self.used += int(getattr(comp, "prompt_tokens", 0) or 0) + int(getattr(comp, "completion_tokens", 0) or 0)
        return comp


def run_session(task: Task, cfg: SessionConfig, client, out_dir: str | Path = "data/traces") -> Path:
    session_id = f"{task.id}-{cfg.condition}-{cfg.seed}-{uuid.uuid4().hex[:6]}"
    client = _BudgetedClient(client, cfg.token_ceiling)
    path = Path(out_dir) / f"{session_id}.jsonl"

    extra = {"n_learners": cfg.n_learners}
    if cfg.is_studio:
        tau, tau_source = cfg.resolve_tau(task)
        extra.update({"facilitator_enabled": cfg.facilitator_enabled, "tau": float(tau),
                      "tau_source": tau_source, "k_proposals": cfg.k_proposals,
                      "convergence_window": CONVERGENCE_WINDOW})
        if cfg.reference_n is not None:
            extra["reference_n"] = cfg.reference_n
    with TraceWriter(path, session_id=session_id, task_id=task.id, condition=cfg.condition,
                     header=_header(client, cfg, extra)) as tr:
        try:
            if cfg.is_studio:
                _run_studio(task, cfg, client, tr, tau=tau, tau_source=tau_source)
            else:
                _run_tutor(task, cfg, client, tr)
            tr.end_note = f"completed; tokens_used={client.used}"
        except TokenBudgetExceeded as e:
            tr.end_note = f"stopped: {e}; tokens_used={client.used}"
    return path


# --------------------------------------------------------------------- Learners (all conditions)
def _run_learners(task: Task, cfg: SessionConfig, client, tr: TraceWriter) -> list[tuple[str, str]]:
    """Each simulated learner takes a turn; a later learner sees the earlier learners' turns (they
    are working together). Returns [(persona_id, text), ...] for the solver to see.

    Arush - fix 2026-10-07 (docs/DEV_LOG.md): the learner turns used to be logged but NEVER shown
    to the Tutor or the Innovator, so the solver got identical input in A and C (and in B and D)
    and the learner factor could not have any effect. They are now passed on.
    """
    turns: list[tuple[str, str]] = []
    for i, persona in enumerate(_pick_personas(cfg), start=1):
        user = task.agent_view
        if turns:
            user += "\n\nWhat the other learner said:\n" + "\n".join(f"- {t}" for _, t in turns)
        user += "\n\nSay in two sentences how you would start."
        msg = [{"role": "system", "content": PROMPTS["learner_base"].format(persona=persona["text"])},
               {"role": "user", "content": user}]
        l = client.chat(msg, seed=cfg.seed + i)
        tr.log(f"learner_{i}", "turn", content=l.text, prompt_tokens=l.prompt_tokens,
               completion_tokens=l.completion_tokens, request_hash=l.request_hash)
        turns.append((persona.get("id", f"L{i}"), l.text.strip()))
    return turns


def _learner_block(turns: list[tuple[str, str]]) -> str:
    if not turns:
        return ""
    lines = "\n".join(f"- Learner {i} ({pid}): {t}" for i, (pid, t) in enumerate(turns, start=1))
    return "\n\nThe learner(s) you are working with said:\n" + lines


# --------------------------------------------------------------------- Tutor (A, C)
def _run_tutor(task: Task, cfg: SessionConfig, client, tr: TraceWriter) -> None:
    learners = _run_learners(task, cfg, client, tr)

    t = client.chat([{"role": "system", "content": PROMPTS["tutor"]},
                     {"role": "user", "content": task.agent_view + _learner_block(learners)}], seed=cfg.seed)
    code = extract_code(t.text)
    tr.log("tutor", "implementation", content=code, prompt_tokens=t.prompt_tokens,
           completion_tokens=t.completion_tokens, request_hash=t.request_hash, proposal_id="p1")
    res = runner.run(code, task.public_tests, backend=cfg.sandbox_backend)
    tr.log("verifier", "verification", **res.as_trace_fields("public"))


# --------------------------------------------------------------------- Studio (B, D, D_minus)
def _run_studio(task: Task, cfg: SessionConfig, client, tr: TraceWriter, *, tau: float, tau_source: str) -> None:
    # simulated-learner contributions, same as the tutor conditions; the Innovator sees them
    learners = _run_learners(task, cfg, client, tr)
    learner_ctx = _learner_block(learners)

    fac = Facilitator(tau=tau, window=CONVERGENCE_WINDOW, enabled=cfg.facilitator_enabled)
    strategies: list[str] = []
    codes: list[str] = []
    results: list = []                  # public-test result per proposal (shown to the selector)
    embeddings: list[np.ndarray] = []
    pending: str | None = None          # a Facilitator instruction to prepend to the next Innovator turn

    # ---- divergence: K rounds of Innovator -> Engineer -> Verifier, with the Facilitator watching
    for k in range(cfg.k_proposals):
        prior = "\n".join(f"- {s}" for s in strategies) or "(none yet)"
        inn_user = (task.agent_view + learner_ctx + f"\n\nStrategies proposed so far:\n{prior}"
                    + (f"\n\nFacilitator: {pending}" if pending else ""))
        pending = None                  # an instruction applies to the next proposal only
        inn = client.chat([{"role": "system", "content": PROMPTS["innovator"]},
                           {"role": "user", "content": inn_user}], seed=cfg.seed + 100 + k)
        strategy = inn.text.strip()
        pid = f"p{k}"
        tr.log("innovator", "proposal", content=strategy, proposal_id=pid,
               prompt_tokens=inn.prompt_tokens, completion_tokens=inn.completion_tokens,
               request_hash=inn.request_hash)

        eng = client.chat([{"role": "system", "content": PROMPTS["engineer"]},
                           {"role": "user", "content": task.agent_view + "\n\nImplement exactly this strategy:\n" + strategy}],
                          seed=cfg.seed + 200 + k)
        code = extract_code(eng.text)
        tr.log("engineer", "implementation", content=code, proposal_id=pid, parent_ids=[pid],
               prompt_tokens=eng.prompt_tokens, completion_tokens=eng.completion_tokens,
               request_hash=eng.request_hash)

        res = runner.run(code, task.public_tests, backend=cfg.sandbox_backend)
        tr.log("verifier", "verification", proposal_id=pid, **res.as_trace_fields("public"))

        emb = _embed(code)
        strategies.append(strategy)
        codes.append(code)
        results.append(res)
        if emb is not None:
            embeddings.append(emb)
            fac.observe(emb)
            fire, metric = fac.should_intervene()
            if fire:
                # Arush - 2026-10-09: name what the proposals converged on, so the prompt is
                # specific ("do NOT use a set") instead of generic (docs/DEV_LOG.md).
                recent = codes[-CONVERGENCE_WINDOW:]
                kind, args = fac.intervene(
                    construct=dominant_construct(recent),
                    approach=approach_name(strategies[-1]),
                    last_two=(approach_name(strategies[-2]), approach_name(strategies[-1])) if len(strategies) >= 2 else None,
                )
                tr.log("facilitator", "intervention", intervention_type=kind,
                       trigger_metric=float(metric), threshold=float(fac.tau),
                       content=f"convergence detected (window mean {metric:.3f} < tau {fac.tau:.3f}); "
                               f"tau_source={tau_source}; instruction: "
                               + PROMPTS[f"facilitator_{kind}"].format(**args).strip())
                pending = PROMPTS[f"facilitator_{kind}"].format(**args)

    if not strategies:
        return

    # ---- selection: the Critic chooses which proposal to develop (all K shown with code + result)
    # Arush - 2026-10-09 (docs/DEV_LOG.md): nothing used to choose between the proposals; the
    # critique always went to the LAST one. Now the Critic sees every proposal and picks one. If its
    # reply has no valid "SELECTED: pN" line, the proposal passing the most public tests is taken
    # (earliest on a tie) and the trace says so. The family clustering is still logged here too.
    fam_note = ""
    if len(embeddings) >= 2:
        from macs.analysis.aggregate import CLUSTER_THRESHOLD as AGG_THRESHOLD, _standardise_for_clustering
        Xc, _ = _standardise_for_clustering(np.vstack(embeddings))
        labels = cluster_families(Xc, distance_threshold=AGG_THRESHOLD)
        fam_note = f"{len(set(labels.tolist()))} solution families among {len(embeddings)} proposals"
    ids = [f"p{k}" for k in range(len(strategies))]
    listing = "\n\n".join(
        f"[{pid}] Strategy: {strategies[k]}\nExample tests: {results[k].tests_passed} passed, "
        f"{results[k].tests_failed} failed{' (crashed or timed out)' if results[k].error else ''}\n"
        f"```python\n{codes[k]}\n```"
        for k, pid in enumerate(ids))
    sel = client.chat([{"role": "system", "content": PROMPTS["critic_selection"]},
                       {"role": "user", "content": task.agent_view + "\n\nProposals:\n\n" + listing}],
                      seed=cfg.seed + 250)
    chosen, how = _parse_selection(sel.text, ids), "critic"
    if chosen is None:
        chosen, how = ids[_best_by_public_tests(results)], "fallback:most_public_tests_passed"
    sel_idx = ids.index(chosen)
    tr.log("critic", "selection", proposal_id=chosen,
           content=f"selected {chosen} ({how}); {fam_note}\n{sel.text.strip()}".strip(),
           prompt_tokens=sel.prompt_tokens, completion_tokens=sel.completion_tokens,
           request_hash=sel.request_hash)

    # ---- bounded critique-revision on the selected proposal (1 to max_revisions rounds)
    # Stops early once a revision passes every public test. The Critic sees the code it critiques
    # and the Engineer sees the code it revises (fix 2026-10-07).
    cur_id, cur_code, cur_res = chosen, codes[sel_idx], results[sel_idx]
    strategy = strategies[sel_idx]
    for r in range(1, cfg.max_revisions + 1):
        crit = client.chat([{"role": "system", "content": PROMPTS["critic"]},
                            {"role": "user", "content": task.agent_view
                             + "\n\nStrategy:\n" + strategy
                             + "\n\nImplementation:\n```python\n" + cur_code + "\n```"
                             + f"\n\nExample tests: {cur_res.tests_passed} passed, {cur_res.tests_failed} failed"
                             + (" (crashed or timed out)" if cur_res.error else "")}],
                           seed=cfg.seed + 300 + r - 1)
        tr.log("critic", "critique", content=crit.text.strip(), proposal_id=cur_id,
               prompt_tokens=crit.prompt_tokens, completion_tokens=crit.completion_tokens,
               request_hash=crit.request_hash)
        rev = client.chat([{"role": "system", "content": PROMPTS["engineer"]},
                          {"role": "user", "content": task.agent_view
                           + "\n\nYour current solution:\n```python\n" + cur_code + "\n```"
                           + "\n\nRevise it to address this critique:\n" + crit.text.strip()}],
                         seed=cfg.seed + 400 + r - 1)
        rcode = extract_code(rev.text)
        rid = f"p_rev{r}"
        tr.log("engineer", "revision", content=rcode, proposal_id=rid, parent_ids=[cur_id],
               prompt_tokens=rev.prompt_tokens, completion_tokens=rev.completion_tokens,
               request_hash=rev.request_hash)
        res = runner.run(rcode, task.public_tests, backend=cfg.sandbox_backend)
        tr.log("verifier", "verification", proposal_id=rid, **res.as_trace_fields("public"))
        cur_id, cur_code, cur_res = rid, rcode, res
        if _passes_all(res, task):
            break


_SELECTED = re.compile(r"SELECTED:\s*\[?\s*(p\d+)\s*\]?", re.I)


def _parse_selection(text: str, ids: list[str]) -> str | None:
    m = _SELECTED.search(text or "")
    if not m:
        return None
    pid = m.group(1).lower()
    return pid if pid in ids else None


def _passes_all(res, task: Task) -> bool:
    return res.error is None and res.tests_failed == 0 and res.tests_passed >= len(task.public_tests)


def _best_by_public_tests(results) -> int:
    """Index of the result with the most public tests passed (no crash preferred); earliest on ties."""
    return max(range(len(results)), key=lambda k: (results[k].error is None, results[k].tests_passed, -k))
