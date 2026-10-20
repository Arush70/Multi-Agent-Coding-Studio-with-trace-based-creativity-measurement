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
pip install -e ".[dev,embed,ui]" REM embed = torch + transformers (Facilitator); ui = dashboard
copy config\models.example.yaml config\models.yaml   REM then edit
pytest
```

Local development uses [Ollama](https://ollama.com) with an 8B Qwen coder checkpoint; the main
experiment uses a 14B-class Apache-2.0 checkpoint on free cloud GPU quota. Both are reached
through the same OpenAI-compatible interface; only `config/models.yaml` changes.

## Workflow

Ollama must be running (`ollama serve`). Every step can be stopped and re-run; it continues where
it stopped.

**From the dashboard** (`streamlit run macs/ui/app.py`): choose the model profile and sandbox in
the sidebar, then References (status, clear, generate, Facilitator check) -> Multi-run (tasks,
conditions, number of seeds; progress and Stop) -> Overview (Analyse all traces) -> Statistics.
SC3 labelling and calibration are in the Labelling tab, the rating study (SC4) and AI judge in
Rating study, the search study on hard and open problems in Open problems, and environment checks
in Setup. Long steps run in the background (logs in `data/jobs/`), so closing the page does not
stop them.

**Conditions** (`macs/conditions.py`): the core A, B, C, D, D_minus, and the extensions E and
E_minus (neutral Innovator, Facilitator on/off), S (tutor sampled to the studio's token budget) and
P (one prompt for five different approaches). **Tasks:** 24 main tasks, 3 contamination-check
tasks (EvoEval), and open problems for the search study (`--tasks open`).

**From the terminal**, the same steps:

```
# 1. reference sets: 100 ordinary solutions per task (novelty + Facilitator threshold)
python scripts/build_reference.py --tasks all --n 100
python scripts/tau_diagnostics.py --representation unixcoder   # Facilitator firing check

# 2. sessions: tasks x conditions x seeds (proposal: 6 seeds per cell)
python scripts/run_experiment.py --tasks all --conditions all --seeds 1-6 --dry-run   # check the plan
python scripts/run_experiment.py --tasks all --conditions all --seeds 1-6             # --sandbox docker for reported numbers
python scripts/run_experiment.py --tasks main --seeds 1-3 --profile llama31_8b,gemma3_12b   # cross-model panel

# 3. results table (also saved to data/results/summary.json for the dashboard)
python scripts/analyze.py
python scripts/analyze.py --representation unixcoder      # parallel representation
python scripts/analyze.py --all-models && python scripts/model_consistency.py

# 4. S8 statistics: planned contrasts, 95% CIs, Holm, minimum detectable effect
python scripts/stats_report.py --tasks 24 --seeds 6
python scripts/stats_report.py --task-set contamination_check
python scripts/frontier.py --plot                          # quality-diversity frontier, Tier 3 payoff
python scripts/mutation_test.py                            # strength of the held-out tests

# 5. rating study (SC4) and the AI judge
python scripts/rating_study.py build && python scripts/rating_study.py export
python scripts/rating_study.py import responses.xlsx && python scripts/rating_study.py judge
python scripts/rating_study.py analyse

# 6. hard and open problems: random vs single vs studio at an equal budget
python scripts/search_open.py run --tasks t3_circle_packing,t3_bin_packing,t3_cap_set --budget 40 --seeds 1-3
python scripts/search_open.py report

# optional: reference-size sensitivity (build --n 200 on some tasks first)
python scripts/reference_sensitivity.py

# dashboard
streamlit run macs/ui/app.py
```

Traces from older versions of the session code are skipped by the analysis
(`--include-old` keeps them). Changes and open decisions are logged in `docs/DEV_LOG.md`.

## Licence

Code: MIT. Data and traces: CC-BY 4.0.
