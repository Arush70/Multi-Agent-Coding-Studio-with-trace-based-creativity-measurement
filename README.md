# Multi-Agent Coding Studio with Trace-Based Creativity Measurement

A studio of role-prompted LLM agents — Innovator, Engineer, Critic, Verifier, and a
Facilitator that intervenes when ideas converge — that solves programming tasks alongside
simulated learners, and a measurement pipeline that scores every proposal in the session
trace for novelty, diversity and correctness. Builds on
[ASTRA](https://github.com/solex2006/ASTRA-social-multi-agent-tutor) (Oyelere, 2026; MIT)
and its [synthetic benchmark](https://doi.org/10.5281/zenodo.18114909) (CC-BY).

## Layout

| Directory | Purpose | Status |
|---|---|---|
| `macs/llm/` | OpenAI-compatible client; endpoint in config; every response cached by request hash | planned |
| `macs/trace/` | Extended ASTRA JSONL trace schema and validator | planned |
| `macs/sandbox/` | Isolated execution of generated code (no network, CPU/memory limits) | planned |
| `macs/tasks/` | Task-bank loader; task format with public and held-out tests | planned |
| `macs/metrics/` | Normalisation, representations, novelty, diversity, usefulness | planned |
| `macs/prompts/` | Role prompts as YAML (versioned; hashes logged) | planned |
| `macs/facilitator.py` | Convergence detection and four interventions | planned |
| `macs/orchestrator.py` | Session loop for cells A, B, C, D, D- | planned |
| `tasks/` | The task bank: `tier1/` and `tier2/` | planned |
| `data/` | Reference distributions, traces, response cache (not committed) | — |
| `docs/` | Pre-registration, ethics application, rater guide | planned |

## Setup

```
py -m venv .venv
.venv\Scripts\activate          REM Windows;  source .venv/bin/activate on Mac/Linux
pip install -e ".[dev]"
copy config\models.example.yaml config\models.yaml   REM then edit
pytest
```

Local development uses [Ollama](https://ollama.com) with an 8B Qwen coder checkpoint; the main
experiment uses a 14B-class Apache-2.0 checkpoint on free cloud GPU quota. Both are reached
through the same OpenAI-compatible interface; only `config/models.yaml` changes.

## Licence

Code: MIT. Data and traces: CC-BY 4.0.
