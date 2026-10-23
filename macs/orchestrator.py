"""Session orchestrator: the loop that turns model calls into a trace.

Takes a task, runs a session in one of the conditions defined in macs/conditions.py, and writes
every step to a trace. Four kinds of solver:

    tutor          A, C      one solution
    studio         B, D, D_minus, E, E_minus
                             Innovator -> Engineer -> Verifier loop, Critic + Facilitator
    sampling       S         the tutor sampled repeatedly up to the studio's token budget
    single_prompt  P         one call asked for K different approaches, each with code

In the studio the Innovator proposes several strategies, the Engineer implements each, the
Verifier runs the sandbox, the Facilitator watches for convergence and pushes for a fresh idea,
and the Critic selects one proposal for bounded critique and revision. D_minus runs the same
studio with the Facilitator switched off, which isolates the Facilitator's effect.

Orchestration is plain Python on purpose: every control-flow decision the results depend on is
visible in this file. The sandbox is run on task.public_tests only, so held-out tests can never
reach a prompt.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from macs import conditions as conds
from macs.llm.client import prompt_hash
from macs.sandbox import runner
from macs.tasks.bank import Task
from macs.trace.writer import TraceWriter
from macs.facilitator import Facilitator, tau_from_reference_sequences, dominant_construct, approach_name
from macs.metrics.represent import ast_features, normalise
from macs.metrics.diversity import cluster_families

PROMPTS = yaml.safe_load((Path(__file__).parent / "prompts" / "roles.yaml").read_text(encoding="utf-8"))
PERSONAS = yaml.safe_load((Path(__file__).parent / "prompts" / "personas.yaml").read_text(encoding="utf-8"))["personas"]

# Bumped when the session logic changes in a way that makes older traces incomparable. It is
# stamped into every trace header; the analysis skips traces from older versions by default
# (scripts/analyze.py --include-old keeps them). History of changes: docs/DEV_LOG.md.
ORCHESTRATOR_VERSION = "0.4.0"
# Oldest version whose traces are still comparable. 0.4.0 moved the second learner level from 2
# to 6 (plus 4 for the curve) and requires an 8192-token server context (the 4096 context cut
# some Critic prompts short), so every earlier trace is archived (scripts/archive_traces.py).
COMPATIBLE_SINCE = "0.4.0"


def is_comparable(version: str | None) -> bool:
    """True when a trace from this orchestrator version can be analysed with current traces."""
    def tup(v):
        try:
            return tuple(int(x) for x in str(v).split("."))
        except ValueError:
            return (0,)
    return tup(version or "0.1.0") >= tup(COMPATIBLE_SINCE)

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
# Space in which the Facilitator measures convergence: "ast" (26 structural features) or
# "unixcoder" (embedding; needs torch and transformers). Chosen from the pilot by the rule in
# docs/PREREGISTRATION.md (2026-10-15): UniXcoder at the 25th percentile. Frozen.
FACILITATOR_REPRESENTATION = "unixcoder"
# Sent once when the Critic's selection reply has no valid "SELECTED: <id>" line.
SELECTION_REMINDER = ("Reply with exactly one line, SELECTED: <id>, naming one of {ids}. "
                      "Do not write code.")
TAU_RULE = "p25_window_mean_on_reference_sequences"
# Baseline S: the tutor is sampled at this temperature, with a new seed each time, until the
# session has used SAMPLING_TOKEN_TARGET tokens (the pilot's mean studio session, D, rounded) or
# made SAMPLING_MAX solutions. Pre-registered 2026-10-20.
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOKEN_TARGET = 17_000
SAMPLING_MAX = 30

CODE_BLOCK = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*?)```", re.S | re.I)
OPEN_BLOCK = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*)", re.S | re.I)


def extract_code(text: str) -> str:
    """The first fenced code block; a block that is opened but never closed (a long reply cut
    off, or one that never closes the fence) runs to the end of the reply; no fence at all
    means the whole reply is taken as code."""
    m = CODE_BLOCK.search(text) or OPEN_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def _embed(code: str, representation: str = "ast") -> np.ndarray | None:
    # Represent a solution in the Facilitator's space so it can measure how close proposals
    # are. Code that doesn't parse returns None and is skipped.
    try:
        normed = normalise(code)
    except Exception:  # noqa: BLE001
        return None
    if representation == "unixcoder":
        from macs.metrics.represent import embed
        return embed([normed])[0]
    try:
        return ast_features(normed)
    except Exception:  # noqa: BLE001
        return None


def check_facilitator_dependencies(representation: str = FACILITATOR_REPRESENTATION) -> str | None:
    """Return a message if the Facilitator's space cannot be computed here, else None."""
    if representation != "unixcoder":
        return None
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return ("The Facilitator measures in UniXcoder space, which needs torch and transformers:\n"
                "    pip install torch transformers\n"
                "(the model, microsoft/unixcoder-base, downloads on first use).")
    return None


_REFERENCE_CACHE: dict[tuple[str, str, str | None], np.ndarray] = {}


def _reference_matrix(task_id: str, representation: str, model: str | None = None) -> np.ndarray:
    """The task's frozen reference in the Facilitator's space, computed once per process
    (embedding 100 solutions takes a few seconds, and a batch run starts many sessions)."""
    key = (task_id, representation, model)
    if key not in _REFERENCE_CACHE:
        from macs.metrics.reference import build_reference_stats
        rep = "embed" if representation == "unixcoder" else "ast"
        _REFERENCE_CACHE[key] = build_reference_stats(task_id, representation=rep, standardise=False,
                                                      model=model).X_ref
    return _REFERENCE_CACHE[key]


@dataclass
class SessionConfig:
    condition: str                    # see macs/conditions.py
    seed: int
    sandbox_backend: str = "docker"   # real runs use docker; dev on Windows uses "subprocess"
    n_learners: int = 1
    k_proposals: int = K_PROPOSALS
    tau: float | None = None          # None -> resolved from the reference set, else DEFAULT_TAU
    token_ceiling: int = TOKEN_CEILING
    max_revisions: int = MAX_REVISIONS
    facilitator_representation: str = FACILITATOR_REPRESENTATION
    reference_n: int | None = field(default=None, init=False)   # size of the reference used for tau
    model: str | None = field(default=None, init=False)         # set by run_session: whose reference to use

    def __post_init__(self) -> None:
        self.spec = conds.get(self.condition)
        self.n_learners = self.spec.n_learners

    @property
    def is_studio(self) -> bool:
        return self.spec.is_studio

    @property
    def facilitator_enabled(self) -> bool:
        return self.spec.facilitator

    @property
    def scaffold(self) -> bool:
        """Studio Innovator sees the earlier strategies and is told to differ (off in E, E_minus)."""
        return self.spec.scaffold

    def resolve_tau(self, task: Task) -> tuple[float, str]:
        """Return (tau, tau_source). Uses the pre-registered reference rule when the task has a
        frozen reference set; otherwise a provisional default. tau_source is written to the trace
        header so every session records where its threshold came from.

        tau is computed in the same space as the Facilitator (FACILITATOR_REPRESENTATION, cosine
        distance) and with the same statistic (mean nearest-earlier-proposal distance over the
        window), on random sequences of K reference solutions; see tau_from_reference_sequences.
        If that is still 0 the Facilitator could never fire; the proposal requires a
        non-degenerate firing rate, so the documented default is used and the trace records it.
        A missing torch/transformers install for the UniXcoder space is an error, not a fallback.
        """
        if self.tau is not None:
            return self.tau, "override"
        from macs.metrics.reference import reference_path
        if not reference_path(task.id, self.model).exists():
            return DEFAULT_TAU, "default:no_reference"
        try:
            X_ref = _reference_matrix(task.id, self.facilitator_representation, self.model)
        except (ImportError, ModuleNotFoundError):
            raise
        except Exception:  # noqa: BLE001 - empty or unparseable reference file
            return DEFAULT_TAU, "default:no_reference"
        self.reference_n = int(X_ref.shape[0])
        tau = tau_from_reference_sequences(X_ref, k=self.k_proposals, window=CONVERGENCE_WINDOW)
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

    Cached responses count too (they are the same tokens), so a re-run from the cache stops at
    exactly the same point.
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
    cfg.model = getattr(client.profile, "model", None)
    path = Path(out_dir) / f"{session_id}.jsonl"

    extra = {"n_learners": cfg.n_learners, "solver": cfg.spec.solver}
    if cfg.spec.solver == "sampling":
        extra.update({"sampling_temperature": SAMPLING_TEMPERATURE,
                      "sampling_token_target": SAMPLING_TOKEN_TARGET, "sampling_max": SAMPLING_MAX})
    if cfg.spec.solver == "single_prompt":
        extra["k_proposals"] = cfg.k_proposals
    if cfg.is_studio:
        tau, tau_source = cfg.resolve_tau(task)
        extra.update({"facilitator_enabled": cfg.facilitator_enabled, "scaffold": cfg.scaffold, "tau": float(tau),
                      "tau_source": tau_source, "tau_rule": TAU_RULE,
                      "facilitator_representation": cfg.facilitator_representation, "k_proposals": cfg.k_proposals,
                      "convergence_window": CONVERGENCE_WINDOW})
        if cfg.reference_n is not None:
            extra["reference_n"] = cfg.reference_n
    with TraceWriter(path, session_id=session_id, task_id=task.id, condition=cfg.condition,
                     header=_header(client, cfg, extra)) as tr:
        try:
            if cfg.is_studio:
                _run_studio(task, cfg, client, tr, tau=tau, tau_source=tau_source)
            elif cfg.spec.solver == "sampling":
                _run_sampling(task, cfg, client, tr)
            elif cfg.spec.solver == "single_prompt":
                _run_single_prompt(task, cfg, client, tr)
            else:
                _run_tutor(task, cfg, client, tr)
            tr.end_note = f"completed; tokens_used={client.used}"
        except TokenBudgetExceeded as e:
            tr.end_note = f"stopped: {e}; tokens_used={client.used}"
    return path


# --------------------------------------------------------------------- Learners (all conditions)
def _run_learners(task: Task, cfg: SessionConfig, client, tr: TraceWriter) -> list[tuple[str, str]]:
    """Each simulated learner takes a turn; a later learner sees the earlier learners' turns (they
    are working together). Returns [(persona_id, text), ...]; the turns are passed to the Tutor
    and to the Innovator, which is what lets the learner factor affect the solver.
    """
    turns: list[tuple[str, str]] = []
    for i, persona in enumerate(_pick_personas(cfg), start=1):
        user = task.agent_view
        if turns:
            who = "the other learner" if len(turns) == 1 else "the other learners"
            user += f"\n\nWhat {who} said:\n" + "\n".join(f"- {t}" for _, t in turns)
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


# --------------------------------------------------------------------- Baseline S (sampling)
def _run_sampling(task: Task, cfg: SessionConfig, client, tr: TraceWriter) -> None:
    """The tutor, sampled again and again at temperature 1.0 with a new seed each time, until the
    session has used the studio's token budget. Same prompt as the tutor conditions, so the only
    difference from C is the number of tries; the only difference from the studio is that the
    tries are independent instead of coordinated."""
    learners = _run_learners(task, cfg, client, tr)
    messages = [{"role": "system", "content": PROMPTS["tutor"]},
                {"role": "user", "content": task.agent_view + _learner_block(learners)}]
    for i in range(1, SAMPLING_MAX + 1):
        if client.used >= SAMPLING_TOKEN_TARGET:
            break
        t = client.chat(messages, seed=cfg.seed + 500 + i, temperature=SAMPLING_TEMPERATURE)
        code = extract_code(t.text)
        pid = f"s{i}"
        tr.log("tutor", "implementation", content=code, proposal_id=pid, prompt_tokens=t.prompt_tokens,
               completion_tokens=t.completion_tokens, request_hash=t.request_hash)
        res = runner.run(code, task.public_tests, backend=cfg.sandbox_backend)
        tr.log("verifier", "verification", proposal_id=pid, **res.as_trace_fields("public"))


# --------------------------------------------------------------------- Baseline P (single prompt)
def split_approaches(text: str) -> list[tuple[str, str]]:
    """Split a single-prompt reply into [(strategy, code), ...], one per fenced code block. The
    strategy is the prose between the previous block (or the start) and this block."""
    out, pos = [], 0
    for m in CODE_BLOCK.finditer(text):
        prose = text[pos:m.start()].strip()
        out.append((prose[-600:], m.group(1).strip()))
        pos = m.end()
    # a final block that was opened but never closed (a long reply cut off) still counts
    tail = OPEN_BLOCK.search(text, pos)
    if tail:
        out.append((text[pos:tail.start()].strip()[-600:], tail.group(1).strip()))
    return out


def _run_single_prompt(task: Task, cfg: SessionConfig, client, tr: TraceWriter) -> None:
    """One model call asked for K genuinely different approaches, each with its own code. The
    strongest single-agent competitor to the studio: it asks for diversity directly, with no team,
    no Facilitator and no critique."""
    learners = _run_learners(task, cfg, client, tr)
    r = client.chat([{"role": "system", "content": PROMPTS["single_prompt"].format(k=cfg.k_proposals)},
                     {"role": "user", "content": task.agent_view + _learner_block(learners)}],
                    seed=cfg.seed + 600)
    parts = split_approaches(r.text)[:cfg.k_proposals]
    for k, (strategy, code) in enumerate(parts):
        pid = f"p{k}"
        # token counts belong to the one call; they are recorded on the first proposal only
        tok = {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
               "request_hash": r.request_hash} if k == 0 else {"request_hash": r.request_hash}
        tr.log("tutor", "proposal", content=strategy or f"approach {k + 1}", proposal_id=pid, **tok)
        tr.log("tutor", "implementation", content=code, proposal_id=pid, parent_ids=[pid])
        res = runner.run(code, task.public_tests, backend=cfg.sandbox_backend)
        tr.log("verifier", "verification", proposal_id=pid, **res.as_trace_fields("public"))
    if not parts:
        tr.log("tutor", "proposal", content=r.text.strip()[:2000] or "(empty reply)", proposal_id="p0",
               prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens, request_hash=r.request_hash)


# --------------------------------------------------------------------- Studio (B, D, D_minus, E, E_minus)
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
        if cfg.scaffold:
            prior = "\n".join(f"- {s}" for s in strategies) or "(none yet)"
            inn_user = task.agent_view + learner_ctx + f"\n\nStrategies proposed so far:\n{prior}"
            inn_prompt = PROMPTS["innovator"]
        else:
            # neutral Innovator (E, E_minus): no list of earlier strategies, no instruction to differ
            inn_user = task.agent_view + learner_ctx
            inn_prompt = PROMPTS["innovator_neutral"]
        inn_user += f"\n\nFacilitator: {pending}" if pending else ""
        pending = None                  # an instruction applies to the next proposal only
        inn = client.chat([{"role": "system", "content": inn_prompt},
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

        emb = _embed(code, cfg.facilitator_representation)
        strategies.append(strategy)
        codes.append(code)
        results.append(res)
        if emb is not None:
            embeddings.append(emb)
            fac.observe(emb)
            # no check after the last proposal: an instruction then could not change anything
            fire, metric = fac.should_intervene() if k < cfg.k_proposals - 1 else (False, None)
            if fire:
                # name what the proposals converged on, so the instruction is specific
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
    # If the Critic's reply has no valid "SELECTED: pN" line, the proposal passing the most public
    # tests is taken (earliest on a tie) and the trace says so. The family clustering is logged on
    # the same event.
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
    sel_text, sel_pt, sel_ct = sel.text.strip(), sel.prompt_tokens, sel.completion_tokens
    if chosen is None:
        # one fixed reminder, then the deterministic fallback
        retry = client.chat([{"role": "system", "content": PROMPTS["critic_selection"]},
                             {"role": "user", "content": task.agent_view + "\n\nProposals:\n\n" + listing},
                             {"role": "assistant", "content": sel.text},
                             {"role": "user", "content": SELECTION_REMINDER.format(ids=", ".join(ids))}],
                            seed=cfg.seed + 251)
        chosen, how = _parse_selection(retry.text, ids), "critic_after_reminder"
        sel_text += "\n\n[reminder] " + retry.text.strip()
        sel_pt, sel_ct = sel_pt + retry.prompt_tokens, sel_ct + retry.completion_tokens
    if chosen is None:
        chosen, how = ids[_best_by_public_tests(results)], "fallback:most_public_tests_passed"
    sel_idx = ids.index(chosen)
    tr.log("critic", "selection", proposal_id=chosen,
           content=f"selected {chosen} ({how}); {fam_note}\n{sel_text}".strip(),
           prompt_tokens=sel_pt, completion_tokens=sel_ct,
           request_hash=sel.request_hash)

    # ---- bounded critique-revision on the selected proposal (1 to max_revisions rounds)
    # Stops early once a revision passes every public test. The Critic sees the code it critiques
    # and the Engineer sees the code it revises.
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
