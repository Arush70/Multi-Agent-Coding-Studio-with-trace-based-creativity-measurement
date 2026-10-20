"""The experimental conditions, defined in one place.

Every part of the system (orchestrator, batch runner, analysis, statistics, dashboard) reads the
conditions from here, so adding one is a single change.

Core design (proposal, section 3.1): a 2 x 2 of learners (1, 2) x solver (Tutor, Studio), plus the
Facilitator ablation.

    A        1 learner  + Tutor
    B        1 learner  + Studio (Innovator told to differ, Facilitator on)
    C        2 learners + Tutor
    D        2 learners + Studio (Innovator told to differ, Facilitator on)
    D_minus  2 learners + Studio (Innovator told to differ, Facilitator off)

Extensions (pre-registered 2026-10-20, docs/PREREGISTRATION.md):

    E        2 learners + Studio, NEUTRAL Innovator (no list of earlier strategies, no instruction
             to differ), Facilitator on. Ideas can converge, so the Facilitator gets a real chance.
    E_minus  as E, Facilitator off.
             D, D_minus, E, E_minus form a 2 x 2 inside the studio: diversity scaffold x
             Facilitator. E - E_minus is the Facilitator's effect where convergence happens.
    S        2 learners + Tutor sampled repeatedly (temperature 1.0, new seed each time) until
             the session has used the studio's token budget. Answers "is the studio just more
             tries?" at equal compute.
    P        2 learners + one model call asked for K genuinely different approaches, each with
             code. Answers "is the team needed, or does one diversity prompt do the same?".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    name: str
    n_learners: int
    solver: str                 # "tutor" | "studio" | "sampling" | "single_prompt"
    facilitator: bool = False   # Facilitator switched on (studio only)
    scaffold: bool = True       # studio Innovator sees earlier strategies and is told to differ
    description: str = ""

    @property
    def is_studio(self) -> bool:
        return self.solver == "studio"

    @property
    def group(self) -> str:
        """Coarse label used by the dashboard headline: 'tutor', 'studio' or 'baseline'."""
        return {"tutor": "tutor", "studio": "studio"}.get(self.solver, "baseline")


CONDITIONS: dict[str, Condition] = {c.name: c for c in [
    Condition("A", 1, "tutor", description="1 learner + Tutor (baseline)"),
    Condition("B", 1, "studio", facilitator=True, description="1 learner + Studio"),
    Condition("C", 2, "tutor", description="2 learners + Tutor"),
    Condition("D", 2, "studio", facilitator=True, description="2 learners + Studio"),
    Condition("D_minus", 2, "studio", facilitator=False, description="2 learners + Studio, Facilitator OFF"),
    Condition("E", 2, "studio", facilitator=True, scaffold=False,
              description="2 learners + Studio, neutral Innovator, Facilitator ON"),
    Condition("E_minus", 2, "studio", facilitator=False, scaffold=False,
              description="2 learners + Studio, neutral Innovator, Facilitator OFF"),
    Condition("S", 2, "sampling", description="2 learners + Tutor sampled to the studio's token budget"),
    Condition("P", 2, "single_prompt", description="2 learners + one prompt for K different approaches"),
]}

CORE = ["A", "B", "C", "D", "D_minus"]
ALL = list(CONDITIONS)
STUDIO = [c for c, v in CONDITIONS.items() if v.is_studio]
TUTOR = [c for c, v in CONDITIONS.items() if v.solver == "tutor"]
BASELINES = [c for c, v in CONDITIONS.items() if v.group == "baseline"]


def get(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError:
        raise ValueError(f"unknown condition {name!r}; known: {', '.join(ALL)}") from None
