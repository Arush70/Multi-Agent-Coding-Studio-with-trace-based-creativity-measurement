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
from macs.facilitator import Facilitator, tau_from_reference
from macs.metrics.represent import ast_features, normalise
from macs.metrics.diversity import cluster_families

PROMPTS = yaml.safe_load((Path(__file__).parent / "prompts" / "roles.yaml").read_text())
PERSONAS = yaml.safe_load((Path(__file__).parent / "prompts" / "personas.yaml").read_text())["personas"]

K_PROPOSALS = 5
MAX_REVISIONS = 3
TOKEN_CEILING = 40_000
CONVERGENCE_WINDOW = 3
# Provisional clustering / tau values, used only when a task has no frozen reference set yet
# (e.g. a pilot run). The REAL values come from the reference set + pre-registration; a trace
# records which one was used via the 'threshold' field on each intervention.
CLUSTER_THRESHOLD = 0.30
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

    def __post_init__(self) -> None:
        self.n_learners = 2 if self.condition in ("C", "D", "D_minus") else 1

    @property
    def is_studio(self) -> bool:
        return self.condition in ("B", "D", "D_minus")

    @property
    def facilitator_enabled(self) -> bool:
        return self.condition in ("B", "D")          # OFF only in D_minus

    def resolve_tau(self, task: Task) -> tuple[float, bool]:
        """Return (tau, from_reference). Uses the pre-registered reference rule when the task
        has a frozen reference set; otherwise a provisional default (flagged in the trace)."""
        if self.tau is not None:
            return self.tau, False
        try:
            from macs.metrics.reference import build_reference_stats
            rs = build_reference_stats(task.id)
            return tau_from_reference(rs.X_ref), True
        except Exception:  # noqa: BLE001 - no reference set yet (pilot); use the default
            return DEFAULT_TAU, False


def _header(client, cfg: "SessionConfig") -> dict:
    return {
        **client.profile.identity,
        "seed": cfg.seed,
        "prompt_hashes": {k: prompt_hash(v) for k, v in PROMPTS.items() if isinstance(v, str)},
        "token_ceiling": TOKEN_CEILING,
        "max_revision_rounds": MAX_REVISIONS,
    }


def _pick_personas(cfg: SessionConfig) -> list[dict]:
    start = cfg.seed % len(PERSONAS)
    return [PERSONAS[(start + j) % len(PERSONAS)] for j in range(cfg.n_learners)]


def run_session(task: Task, cfg: SessionConfig, client, out_dir: str | Path = "data/traces") -> Path:
    session_id = f"{task.id}-{cfg.condition}-{cfg.seed}-{uuid.uuid4().hex[:6]}"
    path = Path(out_dir) / f"{session_id}.jsonl"

    with TraceWriter(path, session_id=session_id, task_id=task.id, condition=cfg.condition,
                     header=_header(client, cfg)) as tr:
        if cfg.is_studio:
            _run_studio(task, cfg, client, tr)
        else:
            _run_tutor(task, cfg, client, tr)
    return path


# --------------------------------------------------------------------- Tutor (A, C)
def _run_tutor(task: Task, cfg: SessionConfig, client, tr: TraceWriter) -> None:
    for i, persona in enumerate(_pick_personas(cfg), start=1):
        msg = [{"role": "system", "content": PROMPTS["learner_base"].format(persona=persona["text"])},
               {"role": "user", "content": task.agent_view + "\n\nSay in two sentences how you would start."}]
        l = client.chat(msg, seed=cfg.seed + i)
        tr.log(f"learner_{i}", "turn", content=l.text, prompt_tokens=l.prompt_tokens,
               completion_tokens=l.completion_tokens, request_hash=l.request_hash)

    t = client.chat([{"role": "system", "content": PROMPTS["tutor"]},
                     {"role": "user", "content": task.agent_view}], seed=cfg.seed)
    code = extract_code(t.text)
    tr.log("tutor", "implementation", content=code, prompt_tokens=t.prompt_tokens,
           completion_tokens=t.completion_tokens, request_hash=t.request_hash, proposal_id="p1")
    res = runner.run(code, task.public_tests, backend=cfg.sandbox_backend)
    tr.log("verifier", "verification", **res.as_trace_fields("public"))


# --------------------------------------------------------------------- Studio (B, D, D_minus)
def _run_studio(task: Task, cfg: SessionConfig, client, tr: TraceWriter) -> None:
    # optional simulated-learner contributions, same as the tutor conditions
    for i, persona in enumerate(_pick_personas(cfg), start=1):
        msg = [{"role": "system", "content": PROMPTS["learner_base"].format(persona=persona["text"])},
               {"role": "user", "content": task.agent_view + "\n\nSay in two sentences how you would start."}]
        l = client.chat(msg, seed=cfg.seed + i)
        tr.log(f"learner_{i}", "turn", content=l.text, prompt_tokens=l.prompt_tokens,
               completion_tokens=l.completion_tokens, request_hash=l.request_hash)

    tau, from_ref = cfg.resolve_tau(task)
    fac = Facilitator(tau=tau, window=CONVERGENCE_WINDOW, enabled=cfg.facilitator_enabled)
    strategies: list[str] = []
    embeddings: list[np.ndarray] = []
    pending: str | None = None          # a Facilitator instruction to prepend to the next Innovator turn

    # ---- divergence: K rounds of Innovator -> Engineer -> Verifier, with the Facilitator watching
    for k in range(cfg.k_proposals):
        prior = "\n".join(f"- {s}" for s in strategies) or "(none yet)"
        inn_user = (task.agent_view + f"\n\nStrategies proposed so far:\n{prior}"
                    + (f"\n\nFacilitator: {pending}" if pending else ""))
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
        if emb is not None:
            embeddings.append(emb)
            fac.observe(emb)
            fire, metric = fac.should_intervene()
            if fire:
                kind, args = fac.intervene(
                    families_so_far=[],
                    last_two=(strategies[-2], strategies[-1]) if len(strategies) >= 2 else None,
                )
                tr.log("facilitator", "intervention", intervention_type=kind,
                       trigger_metric=float(metric), threshold=float(fac.tau),
                       content=f"convergence detected (window mean {metric:.3f} < tau {fac.tau:.3f}); "
                               f"tau_from_reference={from_ref}")
                pending = PROMPTS[f"facilitator_{kind}"].format(**args)
            else:
                pending = None

    # ---- selection: cluster the implementations into solution families (logged, not decided here)
    if len(embeddings) >= 2:
        labels = cluster_families(np.vstack(embeddings), distance_threshold=CLUSTER_THRESHOLD)
        tr.log("critic", "selection",
               content=f"{len(set(labels.tolist()))} solution families among {len(embeddings)} proposals")

    # ---- critique + one revision on the final proposal (bounded; keeps the loop finite)
    if strategies:
        crit = client.chat([{"role": "system", "content": PROMPTS["critic"]},
                            {"role": "user", "content": task.agent_view + "\n\nMost recent strategy:\n" + strategies[-1]}],
                           seed=cfg.seed + 300)
        tr.log("critic", "critique", content=crit.text.strip(),
               prompt_tokens=crit.prompt_tokens, completion_tokens=crit.completion_tokens,
               request_hash=crit.request_hash)
        rev = client.chat([{"role": "system", "content": PROMPTS["engineer"]},
                          {"role": "user", "content": task.agent_view + "\n\nRevise your solution addressing this critique:\n" + crit.text.strip()}],
                         seed=cfg.seed + 400)
        rcode = extract_code(rev.text)
        tr.log("engineer", "revision", content=rcode, proposal_id="p_rev", parent_ids=[f"p{cfg.k_proposals-1}"],
               prompt_tokens=rev.prompt_tokens, completion_tokens=rev.completion_tokens,
               request_hash=rev.request_hash)
        res = runner.run(rcode, task.public_tests, backend=cfg.sandbox_backend)
        tr.log("verifier", "verification", proposal_id="p_rev", **res.as_trace_fields("public"))
