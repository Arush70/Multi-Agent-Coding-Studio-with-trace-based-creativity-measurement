# Multi-Agent Coding Studio with Trace-Based Creativity Measurement

A studio of role-prompted LLM agents — Innovator, Engineer, Critic, Verifier, and a
Facilitator that intervenes when ideas converge — that solves programming tasks alongside
simulated learners, and a measurement pipeline that scores every proposal in the session
trace for novelty, diversity and correctness. Builds on
[ASTRA](https://github.com/solex2006/ASTRA-social-multi-agent-tutor) (Oyelere, 2026; MIT)
and its [synthetic benchmark](https://doi.org/10.5281/zenodo.18114909) (CC-BY).

## Prerequisites and requirements

**To look at the results and rescore them** (no GPU, no model server, no internet after setup):

| What you need | Details |
|---|---|
| Computer | Windows 10/11, macOS or Linux, with 8 GB of RAM or more |
| Disk space | about 4 GB free: the zip is 0.7 GB and it unzips to 3.2 GB (about 144,000 files) |
| Python | version 3.10 or 3.11 (both tested), from [python.org](https://www.python.org/downloads/) |
| Internet | only once, during setup, to download the Python packages |
| Python packages | installed by one command in Step 5 below (NumPy, SciPy, scikit-learn, statsmodels, pandas, PyYAML, jsonschema, radon, black, Streamlit, python-docx, openpyxl, matplotlib, pytest) |
| Optional | PyTorch and transformers, only for the extra UniXcoder check (Step 10) |

**Only to generate new traces with the models** (not needed to check the results): Ollama, a GPU
(NVIDIA A100 80 GB for the larger models; full list in `REPRODUCE.md`, section 2), about 200 GB
of disk for the model weights, and internet access to download the models.

## Tutorial: set up, look at the results, and rescore everything

The zip already contains everything: all 12,909 analysed session traces (plus the archived
ones), every model's reference sets, the cache of every model response, the 65,928 stored test
results, and every result file. Nothing has to be generated again, and no model is run.

### Step 1. Install Python

Download Python 3.11 from [python.org](https://www.python.org/downloads/) and install it.
On Windows, tick **"Add python.exe to PATH"** on the first screen of the installer.

To check it worked, open a terminal (Windows: press the Windows key, type `cmd`, press Enter;
Mac: open the **Terminal** app) and type:

    python --version

It should print `Python 3.10.x` or `Python 3.11.x`. (On Windows, if it says "python is not
recognized", type `py` instead of `python` in every command below. On a Mac, use `python3`.)

### Step 2. Unzip the package

Unzip `MACS_dissertation-submission-v1.0.zip` into a folder with a **short path**, for example
`C:\MACS` on Windows (right-click the zip, **Extract All...**, type `C:\MACS`). A short path
matters on Windows because some file names inside are long. Unzipping 144,000 files can take
several minutes.

You now have a folder `C:\MACS\MACS_dissertation-submission-v1.0` containing `README.md`,
`macs`, `scripts`, `data` and so on.

### Step 3. Open a terminal in that folder

- **Windows:** open the folder `MACS_dissertation-submission-v1.0` in File Explorer, click the
  address bar at the top, type `cmd` and press Enter. A black window opens in that folder.
- **Mac/Linux:** open Terminal and type `cd ` followed by the folder path, for example
  `cd ~/Downloads/MACS_dissertation-submission-v1.0`.

### Step 4. Create a private Python environment (once)

This keeps the project's packages separate from the rest of your computer.

    python -m venv .venv

Then switch it on:

- Windows: `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

You will see `(.venv)` at the start of the line. Do this "switch on" step every time you open a
new terminal for this project.

### Step 5. Install the packages (once, needs internet)

    pip install -e ".[dev,ui]" matplotlib

This takes a few minutes. When it finishes without a red error, you are ready.

### Step 6. Check that everything works

    pytest

At the end it should say `135 passed`.

### Step 7. Open the dashboard

    streamlit run macs/ui/app.py

Your web browser opens the dashboard at `http://localhost:8501` (if it does not, open that
address yourself). The first time, Streamlit may ask for an email address in the terminal: just
press Enter. **Keep the terminal window open** while you use the dashboard.

### Step 8. Look at the submitted results

Nothing needs to be run to see the results:

- **Overview** tab: Studio vs Tutor for each model. Choose the model in **Model shown**
  (Qwen3.5-27B is the main model).
- **Statistics** tab: the planned comparisons, confidence intervals and the comparison across
  the eight models.
- **Traces** tab: read any session, message by message.

### Step 9. Rescore everything from the traces

On the **Overview** tab, click **Rescore everything (all models, all results)**.

- A panel appears: **⏳ Rescore everything: running**, with a progress bar that counts 23 steps.
- It rescores every stored trace of every model against the stored reference sets (families,
  novelty, held-out tests) and rebuilds every file in `data/results/`: per-model results and
  statistics, the meta-analysis across models, the contamination check, the frontier and the
  mutation test.
- It took **84 minutes on a 2-core computer**; a computer with more cores is faster. You can
  switch to other tabs or close the browser tab, but keep the terminal window open.
- When the panel shows **✅ Rescore everything: finished**, refresh the page. The tabs now show the
  rescored results, which are identical to the submitted ones.

**Re-analyse <model>'s traces** (the button above it) does the same for just the one model shown.

**Why the results come out identical on any computer:** whether a program passes its tests was
decided on the ISCA cluster by running it under a CPU-time limit. Those 65,928 verdicts are stored
in `data/results/execution/` and rescoring reuses them, so a slower or faster computer gives the
same numbers. To run every program again instead, use `python scripts/rescore_all.py --reexecute`
in the terminal; then a few programs that finish close to the time limit can pass on one computer
and time out on another.

### Step 10 (optional). The UniXcoder check

This repeats the main model's novelty measure with a neural code model instead of the code
structure. It needs PyTorch (a large download) and downloads `microsoft/unixcoder-base` from
Hugging Face the first time:

    pip install -e ".[embed]"

After this, **Rescore everything** includes the UniXcoder check as its last step (it took 61
minutes more on a 2-core computer). Because it uses a neural network, its numbers can differ in
the last digits on different hardware; every other file is identical.

### Step 11. Stop, and start again later

To stop the dashboard, click the terminal window and press `Ctrl+C`. To start again later:
open a terminal in the folder (Step 3), switch the environment on (Step 4), and run
`streamlit run macs/ui/app.py` (Step 7). Steps 5 and 6 are needed only once.

### The same from the terminal, without the dashboard

With the environment switched on (Step 4):

    python scripts/rescore_all.py

This rescores everything, exactly like the dashboard button, and prints its progress.

To recompute **only the statistics** from the stored per-session results (about 5 minutes):

    python scripts/stats_report.py --tasks 24 --seeds 6
    python scripts/model_consistency.py
    python scripts/rating_study.py analyse

Which file holds which result, and what each level recomputes: `REPRODUCE.md`, section 0.

### If something goes wrong

- **"python is not recognized":** use `py` (Windows) or `python3` (Mac) instead of `python`.
- **"pip is not recognized" or packages missing:** the environment is not switched on; repeat
  Step 4 and look for `(.venv)`.
- **Errors while unzipping on Windows:** unzip again into a shorter folder such as `C:\MACS`.
- **The browser does not open:** go to `http://localhost:8501` yourself.
- **"already running" message:** a rescore is still in progress; wait for it to finish, or press
  **Stop** in its panel.

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
