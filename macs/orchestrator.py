"""Session orchestrator - the loop that turns model calls into a trace.

Arush - this is the piece that finally ties everything together: it takes a task, asks
the model to play the learner(s) and the tutor, runs the resulting code through the
sandbox, and writes every step to a trace. When you run it, you get your FIRST real
experimental data file.

The five conditions:
    A        1 learner  + Tutor
    B        1 learner  + studio (Innovator, Engineer, Critic, Verifier, Facilitator)
    C        2 learners + Tutor
    D        2 learners + studio
    D_minus  2 learners + studio, Facilitator disabled

STATUS: conditions A and C (Tutor-only) are wired end to end - enough to produce a real
trace this week and to be your experimental baseline. The studio branch (B, D, D_minus)
is laid out with TODOs for weeks 3-4, when the Innovator/Critic/Facilitator loop is built.
Keeping orchestration as plain Python (not a framework) is deliberate: every control-flow
decision your results depend on is visible and reviewable in one file.

The one rule enforced here: the sandbox is run on task.public_tests ONLY. Held-out tests
never enter this file, so they can never reach a prompt. Scoring on held-out tests is a
separate, later step over the finished traces.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from macs.llm.client import prompt_hash
from macs.sandbox import runner
from macs.tasks.bank import Task
from macs.trace.writer import TraceWriter

PROMPTS = yaml.safe_load((Path(__file__).parent / "prompts" / "roles.yaml").read_text())
PERSONAS = yaml.safe_load((Path(__file__).parent / "prompts" / "personas.yaml").read_text())["personas"]

K_PROPOSALS = 5
MAX_REVISIONS = 3
TOKEN_CEILING = 40_000
# Pulls the code out of a ```python ...``` block; if the model forgot the fence, fall
# back to the whole reply so a stray formatting choice never silently drops a solution.
CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


@dataclass
class SessionConfig:
    condition: str                    # A | B | C | D | D_minus
    seed: int
    sandbox_backend: str = "docker"   # the REAL runs use docker; dev on Windows uses "subprocess"
    n_learners: int = 1

    def __post_init__(self) -> None:
        # Two-learner conditions are C, D, D_minus; the rest are single-learner.
        self.n_learners = 2 if self.condition in ("C", "D", "D_minus") else 1


def _header(client, seed: int) -> dict:
    # Everything needed to reproduce the session: model identity + the hash of EVERY role
    # prompt. If you edit a prompt later, the hash changes and the trace shows it.
    return {
        **client.profile.identity,
        "seed": seed,
        "prompt_hashes": {k: prompt_hash(v) for k, v in PROMPTS.items() if isinstance(v, str)},
        "token_ceiling": TOKEN_CEILING,
        "max_revision_rounds": MAX_REVISIONS,
    }


def run_session(task: Task, cfg: SessionConfig, client, out_dir: str | Path = "data/traces") -> Path:
    # session_id is unique per run (task-condition-seed-random) so parallel runs never
    # collide on a filename.
    session_id = f"{task.id}-{cfg.condition}-{cfg.seed}-{uuid.uuid4().hex[:6]}"
    path = Path(out_dir) / f"{session_id}.jsonl"
    tokens_used = 0

    with TraceWriter(path, session_id=session_id, task_id=task.id, condition=cfg.condition,
                     header=_header(client, cfg.seed)) as tr:
        if cfg.condition in ("A", "C"):
            # --- Tutor-only conditions -------------------------------------------------
            # Pick one persona per learner, deterministically from the seed, so the same
            # seed always yields the same learners (reproducibility).
            start = cfg.seed % len(PERSONAS)
            personas = [PERSONAS[(start + j) % len(PERSONAS)] for j in range(cfg.n_learners)]

            for i, persona in enumerate(personas, start=1):
                learner_msg = [
                    {"role": "system", "content": PROMPTS["learner_base"].format(persona=persona["text"])},
                    {"role": "user", "content": task.agent_view + "\n\nSay in two sentences how you would start."},
                ]
                # seed+i so the two learners in a pair are not identical draws.
                l = client.chat(learner_msg, seed=cfg.seed + i)
                tokens_used += l.prompt_tokens + l.completion_tokens
                tr.log(f"learner_{i}", "turn", content=l.text, prompt_tokens=l.prompt_tokens,
                       completion_tokens=l.completion_tokens, request_hash=l.request_hash)

            # The Tutor produces one solution.
            tutor_msg = [{"role": "system", "content": PROMPTS["tutor"]},
                         {"role": "user", "content": task.agent_view}]
            t = client.chat(tutor_msg, seed=cfg.seed)
            tokens_used += t.prompt_tokens + t.completion_tokens
            code = extract_code(t.text)
            tr.log("tutor", "implementation", content=code, prompt_tokens=t.prompt_tokens,
                   completion_tokens=t.completion_tokens, request_hash=t.request_hash, proposal_id="p1")

            # Verify on PUBLIC tests only. Held-out scoring happens later, over the traces.
            res = runner.run(code, task.public_tests, backend=cfg.sandbox_backend)
            tr.log("verifier", "verification", **res.as_trace_fields("public"))

        else:
            # --- studio conditions: built in weeks 3-4 --------------------------------
            # divergence (Innovator x K, Facilitator.observe/intervene) -> selection
            # (cluster into families) -> implementation (Engineer) -> critique/revision
            # loop (<= MAX_REVISIONS) -> verification on PUBLIC tests. D_minus runs the
            # studio with Facilitator(enabled=False): it observes and logs but never fires.
            raise NotImplementedError("studio conditions (B, D, D_minus) are implemented in weeks 3-4")

        # The token ceiling is a pre-registered stopping rule; log if a session blew past it.
        if tokens_used > TOKEN_CEILING:
            tr.log("system", "turn", content=f"token ceiling exceeded: {tokens_used}")

    return path
