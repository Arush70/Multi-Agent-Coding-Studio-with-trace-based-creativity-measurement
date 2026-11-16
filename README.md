# Multi-Agent Coding Studio with Trace-Based Creativity Measurement

A studio of role-prompted LLM agents — Innovator, Engineer, Critic, Verifier, and a
Facilitator that intervenes when ideas converge — that solves programming tasks alongside
simulated learners, and a measurement pipeline that scores every proposal in the session
trace for novelty, diversity and correctness. Builds on
[ASTRA](https://github.com/solex2006/ASTRA-social-multi-agent-tutor) (Oyelere, 2026; MIT)
and its [synthetic benchmark](https://doi.org/10.5281/zenodo.18114909) (CC-BY).

## Prerequisites and requirements

To check and rescore the results (no GPU, no model server):

| Requirement | Detail |
|---|---|
| Computer | Windows 10/11, Linux or macOS; 8 GB RAM or more; about 3 GB free disk after unzipping |
| Python | 3.10 or later (tested on 3.10 and 3.11), with `pip` and `venv` |
| Python packages | `pip install -e ".[dev,ui]"` installs everything needed (NumPy, SciPy, scikit-learn, statsmodels, pandas, PyYAML, jsonschema, radon, black, Streamlit, python-docx, openpyxl) |
| Optional | `pip install matplotlib` for the frontier chart; `pip install -e ".[embed]"` (PyTorch, transformers) for the UniXcoder check |

Only to generate new traces with the models (not needed to check the results): Ollama, a GPU
(NVIDIA A100 80 GB for the larger models; full list in `REPRODUCE.md`, section 2), about 200 GB
of disk for the model weights, and internet access to download the models.

## Checking and rescoring the results without rerunning the models

The package contains all 12,909 analysed traces (plus the archived ones), every model's reference
sets, the response cache and every result, so no traces or references have to be generated
again.

**Setup (once):**

    cd MACS_dissertation-submission-v1.0
    python -m venv .venv
    .venv\Scripts\activate                 (Linux/Mac: source .venv/bin/activate)
    pip install -e ".[dev,ui]"
    pytest                                  (all tests should pass)

**From the dashboard:** run `streamlit run macs/ui/app.py`, open the **Overview** tab and click
**Rescore everything (all models, all results)**. It rescores every stored trace of every model
against the stored reference sets (families, novelty, held-out tests) and rebuilds every file in
`data/results/`: per-model results and statistics, the meta-analysis across models, the
contamination check, the frontier and the mutation test. It runs in the background with a
progress bar; the tabs then show the rescored results. **Re-analyse <model>'s traces** does the
same for the one model shown.

**From the terminal**, the same in one command:

    python scripts/rescore_all.py

**Only the statistics** (about 5 minutes, from the stored per-session results):

    python scripts/stats_report.py --tasks 24 --seeds 6
    python scripts/model_consistency.py
    python scripts/rating_study.py analyse

Which file holds which result, and what each level recomputes: `REPRODUCE.md`, section 0.

## Documentation

**How everything works, the full prerequisites, how to rerun each part and the compute time:
`REPRODUCE.md`.** Every frozen decision and amendment is dated in `docs/PREREGISTRATION.md`;
the exact compute time of every run is in `docs/COMPUTE_TIME.md`.

## Layout

| Directory | Purpose | Status |
|---|---|---|
| `macs/llm/` | OpenAI-compatible client; endpoint in config; every response cached by request hash | done |
| `macs/trace/` | Extended ASTRA JSONL trace schema and validator | done |
| `macs/sandbox/` | Isolated execution of generated code (no network, CPU/memory limits) | done |
| `macs/tasks/` | Task-bank loader; task format with public and held-out tests | done |
| `macs/metrics/` | Normalisation, representations, novelty, diversity, usefulness | done |
| `macs/prompts/` | Role prompts as YAML (versioned; hashes logged) | done |
| `macs/facilitator.py` | Convergence detection and four interventions | done |
| `macs/orchestrator.py` | Session loop for cells A, B, C, D, D- | done |
| `tasks/` | The task bank: `tier1/`, `tier2/` and `tier3/` | done |
| `data/` | Traces, reference sets, response cache and results (all in the submission package; only results are in git) | done |
| `docs/` | Pre-registration, project notes, development log, compute time | done |

## Setup for new runs with the models

```
py -m venv .venv
.venv\Scripts\activate          REM Windows;  source .venv/bin/activate on Mac/Linux
pip install -e ".[dev,embed,ui]" REM embed = torch + transformers (Facilitator); ui = dashboard
copy config\models.example.yaml config\models.yaml   REM then edit
pytest
```

All models are served by [Ollama](https://ollama.com) through the same OpenAI-compatible
interface; only `config/models.yaml` changes. The reported runs used Qwen3.5-27B as the main model
and seven further open models on the University GPU cluster (`isca/README.md`).

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
