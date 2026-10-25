"""Iterated program search on hard and open problems (FunSearch-style), with three strategies.

Every strategy gets the same budget: the number of programs that are written and scored. Each
program is scored by the task's trusted evaluator (macs.analysis.heuristic_score), where higher is
better and 1.0 is the standard baseline heuristic; an invalid program scores 0.

    random   independent samples from the task alone (temperature 1.0, new seed each time):
             no feedback, the "just more tries" baseline
    single   one model, shown the best programs so far with their scores, asked for a better one
             each step (a single-agent FunSearch-style loop)
    studio   the studio's division of labour: the Innovator proposes a NEW strategy (shown the best
             programs and every strategy tried, told to differ), the Engineer implements it, and
             the Facilitator watches the programs for convergence and steps in (as in the sessions)

The comparison that answers the research question: best score reached vs number of programs
scored, per strategy, averaged over seeds; tokens are logged as the cost. Every program, score
and prompt hash goes to a JSONL log (data/search/<model>/<task>-<strategy>-<seed>.jsonl; one folder
per model, so runs of different models never mix).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from macs.analysis.heuristic_score import score_heuristic_code
from macs.facilitator import Facilitator, code_constructs, dominant_construct
from macs.llm.client import prompt_hash
from macs.orchestrator import extract_code, _embed, DEFAULT_TAU, CONVERGENCE_WINDOW

PROMPTS = yaml.safe_load((Path(__file__).resolve().parents[1] / "prompts" / "search.yaml").read_text(encoding="utf-8"))
STRATEGIES = ("random", "single", "studio")
TEMPERATURE = 1.0
K_TOP = 3                    # best programs shown in a prompt
MAX_SHOWN_CHARS = 2500       # each shown program is cut to this length to bound the prompt


@dataclass
class Program:
    step: int
    code: str
    score: float
    valid: bool
    metric: float | None
    strategy_text: str = ""


def _show(best: list[Program]) -> str:
    if not best:
        return "(no programs yet)"
    return "\n\n".join(f"Score {p.score:.4f}:\n```python\n{p.code[:MAX_SHOWN_CHARS]}\n```" for p in best)


def _score(task, code, backend) -> tuple[float, bool, float | None, str | None]:
    r = score_heuristic_code(task, code, backend=backend)
    s = r.get("score")
    return (float(s) if s is not None else 0.0), bool(r.get("valid")), r.get("metric"), r.get("error")


def run_search(task, client, *, strategy: str, budget: int = 40, seed: int = 1, backend: str = "subprocess",
               out_dir: str | Path = "data/search", log=print) -> Path:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    from macs.metrics.reference import model_slug
    model = getattr(client.profile, "model", None)
    out_dir = Path(out_dir) / model_slug(model or "unknown_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task.id}-{strategy}-{seed}.jsonl"
    logged = read_curve(path)["model"] if path.exists() else None
    if logged is not None and model is not None and logged != model:
        raise RuntimeError(f"{path} is a search by {logged}, not {model}; move it aside before resuming")
    done = _resume(path)
    programs: list[Program] = [Program(**{k: v for k, v in d.items() if k in Program.__annotations__}) for d in done]
    tokens = sum(d.get("tokens", 0) for d in done)
    fac = Facilitator(tau=DEFAULT_TAU, window=CONVERGENCE_WINDOW, enabled=(strategy == "studio"))
    for p in programs:
        emb = _embed(p.code, "ast")
        if emb is not None:
            fac.observe(emb)
    tried = [p.strategy_text for p in programs if p.strategy_text]
    pending = None
    mode = "a" if done else "w"
    with path.open(mode, encoding="utf-8") as fh:
        if not done:
            fh.write(json.dumps({"event": "start", "task_id": task.id, "strategy": strategy, "seed": seed,
                                 "budget": budget, "k_top": K_TOP, "temperature": TEMPERATURE,
                                 "model": getattr(client.profile, "model", None), "revision": getattr(client.profile, "revision", ""),
                                 "prompt_hashes": {k: prompt_hash(v) for k, v in PROMPTS.items() if isinstance(v, str)},
                                 "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        for step in range(len(programs), budget):
            best = sorted(programs, key=lambda p: -p.score)[:K_TOP]
            s = seed * 10_000 + step
            text, used = "", 0
            if strategy == "random":
                r = client.chat([{"role": "system", "content": PROMPTS["writer"]},
                                 {"role": "user", "content": task.agent_view}], seed=s, temperature=TEMPERATURE)
                code, used = extract_code(r.text), r.prompt_tokens + r.completion_tokens
            elif strategy == "single":
                r = client.chat([{"role": "system", "content": PROMPTS["writer"]},
                                 {"role": "user", "content": task.agent_view + "\n\n" + PROMPTS["improve"]
                                  + "\n\n" + _show(best)}], seed=s, temperature=TEMPERATURE)
                code, used = extract_code(r.text), r.prompt_tokens + r.completion_tokens
            else:
                listing = "\n".join(f"- {t[:300]}" for t in tried[-12:]) or "(none yet)"
                user = (task.agent_view + "\n\nBest programs so far:\n" + _show(best)
                        + f"\n\nStrategies already tried:\n{listing}"
                        + (f"\n\nFacilitator: {pending}" if pending else ""))
                pending = None
                inn = client.chat([{"role": "system", "content": PROMPTS["innovator"]},
                                   {"role": "user", "content": user}], seed=s, temperature=TEMPERATURE)
                text = inn.text.strip()
                eng = client.chat([{"role": "system", "content": PROMPTS["engineer"]},
                                   {"role": "user", "content": task.agent_view + "\n\nStrategy:\n" + text
                                    + ("\n\nBest program so far:\n" + _show(best[:1]) if best else "")}],
                                  seed=s + 5_000, temperature=0.7)
                code = extract_code(eng.text)
                used = inn.prompt_tokens + inn.completion_tokens + eng.prompt_tokens + eng.completion_tokens
                tried.append(text)
            score, valid, metric, err = _score(task, code, backend)
            tokens += used
            prog = Program(step=step, code=code, score=score, valid=valid, metric=metric, strategy_text=text)
            programs.append(prog)
            event = {"event": "program", "step": step, "score": score, "valid": valid, "metric": metric,
                     "error": (err or "")[:300] or None, "tokens": used, "tokens_total": tokens,
                     "best_so_far": max(p.score for p in programs), "code": code, "strategy_text": text}
            emb = _embed(code, "ast")
            if strategy == "studio" and emb is not None:
                fac.observe(emb)
                fire, m = fac.should_intervene()
                if fire:
                    recent = [p.code for p in programs[-CONVERGENCE_WINDOW:]]
                    pending = PROMPTS["facilitator"].format(excluded=dominant_construct(recent) or "the same idea")
                    event["intervention"] = {"window_mean": float(m), "tau": float(fac.tau), "instruction": pending}
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            fh.flush()
            if log:
                log(f"[{task.id} {strategy} seed={seed}] step {step + 1}/{budget}: score {score:.4f} "
                    f"(best {event['best_so_far']:.4f})" + ("  [facilitator]" if "intervention" in event else ""))
    return path


def _resume(path: Path) -> list[dict]:
    """Programs already logged in an interrupted run (safe to stop and start again)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") == "program":
            out.append(d)
    return out


def read_curve(path: Path) -> dict:
    """Best-so-far curve and summary of one search log."""
    if not Path(path).exists():
        return {"model": None, "n": 0}
    rows = _resume(Path(path))
    head = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") == "start":
            head = d
            break
    best = list(np.maximum.accumulate([r["score"] for r in rows])) if rows else []
    return {"task_id": head.get("task_id"), "strategy": head.get("strategy"), "seed": head.get("seed"),
            "model": head.get("model"), "n": len(rows), "curve": [float(b) for b in best],
            "best": float(best[-1]) if best else None,
            "best_metric": max((r["metric"] for r in rows if r.get("valid") and r.get("metric") is not None),
                               default=None),
            "valid_share": float(np.mean([r["valid"] for r in rows])) if rows else None,
            "tokens": int(sum(r.get("tokens", 0) for r in rows)),
            "interventions": sum(1 for r in rows if "intervention" in r),
            "distinct_constructs": len({c for r in rows for c in code_constructs(r["code"])})}
