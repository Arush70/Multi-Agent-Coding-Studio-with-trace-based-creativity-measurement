# MACS: how it works and how to reproduce every result

This is the code submitted with the MSc dissertation (COMM514, University of Exeter, December
2026). The submitted version carries the git tag `dissertation-submission-v1.0`; later work for
publication is done after that tag and does not change it. To get exactly that version:

    git checkout dissertation-submission-v1.0

Author: Arush Kumar Vishwakarma. Supervisor: Prof. Solomon S. Oyelere. Code MIT, data CC-BY 4.0.

---

## 0. Checking the results without rerunning anything

The package contains every trace, reference set and intermediate result, so none of the model
runs (GPU) has to be repeated. There are three levels; each needs less than the next.

### Level A: read the results (no installation)

Unzip the package and open these files. Every number in the dissertation comes from one of them.

| Result in the dissertation | File |
|---|---|
| Main model (Qwen3.5-27B): measures per condition | `data/results/summary.json` |
| Main model: planned contrasts, 95% CIs, Holm-corrected tests | `data/results/stats.json` |
| Every model: measures and contrasts | `data/results/models/<model>/summary.json`, `stats.json` |
| Meta-analysis across the 8 models | `data/results/model_consistency.json` |
| Contamination-check tasks | `data/results/stats_contamination.json` |
| UniXcoder representation (check) | `data/results/summary_unixcoder.json` |
| Quality-diversity frontier | `data/results/frontier.json`, `frontier.png` |
| Strength of the held-out tests | `data/results/mutation.json` |
| Reference-size sensitivity | `data/results/reference_sensitivity_qwen3.5_27b.json` |
| Open-problem search | `data/results/search_report.json` |
| Human validation of the novelty measure | `data/rating/results.json` |
| Family check against labels (SC3): labels and agreement | `data/calibration/labels.jsonl`, `data/rating/ai_raters/answers/README.txt` |
| Sessions per model and GPU time | `docs/COMPUTE_TIME.md` |
| Every frozen decision, amendment and result, dated | `docs/PREREGISTRATION.md` |

### Level B: recompute the statistics (about 5 minutes, ordinary laptop, no GPU, no model)

Recomputes every table from the stored per-session measures and the stored ratings and labels.

    cd MACS_dissertation-submission-v1.0
    python -m venv .venv
    .venv\Scripts\activate                 (Linux/Mac: source .venv/bin/activate)
    pip install -e ".[dev]"

    python scripts/stats_report.py --tasks 24 --seeds 6                       # main model contrasts
    python scripts/stats_report.py --task-set contamination_check --out data/results/stats_contamination.json
    python scripts/stats_report.py --summary data/results/models/gemma3_12b/summary.json --tasks 24 --seeds 3 \
        --out data/results/models/gemma3_12b/stats.json                     # any other model (6 seeds for qwen3.5_9b)
    python scripts/model_consistency.py                                      # meta-analysis across models
    python scripts/frontier.py                                               # frontier (--plot needs matplotlib)
    python scripts/rating_study.py analyse                                   # human validation
    python scripts/calibrate_clustering.py                                   # family check (SC3) from the labels
        # prints the ARI sweep (best 0.22 at 5.5); no value reaches 0.60, so the pre-registered 3.0
        # is kept (docs/PREREGISTRATION.md, 2026-11-15)
    python scripts/compute_time.py --sacct data/jobs/sacct_all.txt           # sessions and GPU time

Each command rewrites its file in `data/results/` (or `data/rating/`, `docs/`). The outputs are
identical to the files in the package (checked: `frontier.json` differs only in the 17th
significant digit of two values, from floating-point summation order). To compare, copy
`data/results/` and `data/rating/results.json` somewhere else first.

### Level C: rescore every stored trace (CPU only, no GPU, no model)

Scores every stored session again (normalisation, AST features, families, novelty against the
stored reference sets, held-out tests in the sandbox) for every model, then rebuilds every result
file (per-model statistics, meta-analysis, contamination check, frontier, mutation test).

- **Dashboard:** `pip install -e ".[dev,ui]"`, then `streamlit run macs/ui/app.py`, tab
  **Overview**, button **Rescore everything (all models, all results)**. It runs in the
  background with progress; the tabs show the rescored results when it finishes.
  **Re-analyse <model>'s traces** rescores only the model shown.
- **Terminal:** `python scripts/rescore_all.py` (`--workers N` sets how many models are scored in
  parallel; `--skip-mutation` leaves out the mutation test).
- **Slurm cluster:** `sbatch isca/analyze_all.sbatch` does the same in one job.

The UniXcoder check is included when `pip install -e ".[embed]"` has been run.

Only to generate new traces with the models (GPU) are sections 3 and 4 needed.

---

## 1. What the system does

MACS compares two ways an AI model can help with a programming task:

- **Tutor:** one AI tutor works with simulated learners and proposes solutions.
- **Studio:** a team of role-prompted agents (Innovator proposes different approaches, Engineer
  writes the code, Critic criticises it, Verifier runs the tests, and a Facilitator steps in when
  the ideas start to look alike) works with the same simulated learners.

Every session is written to a **trace** (one JSONL file per session, schema in `macs/trace/`).
A measurement pipeline then scores every solution proposed in the trace:

1. **Represent:** comments and docstrings removed, names normalised, then a 26-number AST
   feature vector per solution (UniXcoder embeddings as a separate check).
2. **Diversity:** solutions of a session are clustered into *families* (agglomerative,
   within-session standardised AST features, `CLUSTER_THRESHOLD = 3.0`); also mean pairwise
   distance, which needs no clustering.
3. **Novelty:** distance to the 5 nearest of 100 ordinary solutions to the same task (the
   *reference set*, sampled once from the same model with a neutral prompt), as a z-score; a
   solution is novel if it is in the top 10%.
4. **Correctness:** hidden held-out tests run in a sandbox (`macs/sandbox/`).
5. **Creativity outcome:** the share of solutions that are both correct and novel.

Results are compared between conditions with a mixed model per model, Holm correction, and a
random-effects meta-analysis across models.

### Conditions (`macs/conditions.py`)

| Code | Meaning |
|---|---|
| A, C | Tutor with 1 or 6 simulated learners |
| B, D | Studio with 1 or 6 simulated learners |
| C4, D4 | Tutor and Studio with 4 learners (learner curve) |
| D_minus | Studio with the Facilitator switched off |
| E, E_minus | Studio with a neutral Innovator, Facilitator on / off |
| S | Tutor sampled until it uses the studio's token budget (equal compute) |
| P | One prompt asking for five different approaches |

### Tasks (`tasks/`)

24 main tasks (MBPP and HumanEval, tiers 1 and 2), 3 contamination-check tasks (EvoEval), and
tier-3 heuristic and open problems used in the search study. Each task folder has the problem,
public tests, held-out tests and the `known_families` approaches written before any model ran.
Sources: `docs/TASK_SOURCES.md`.

### Where things are

| Folder | Contents |
|---|---|
| `macs/` | the system: agents and session loop (`orchestrator.py`, `facilitator.py`), LLM client, trace, sandbox, metrics, analysis, rating study |
| `scripts/` | command-line entry points for every step below |
| `isca/` | job scripts for the University GPU cluster (ISCA) |
| `tasks/` | the task bank |
| `tests/` | unit tests (`pytest`) |
| `config/` | `models.example.yaml` (copy to `models.yaml`; never commit that file) |
| `docs/` | `PREREGISTRATION.md` (every frozen decision and amendment, dated), `PROJECT_MASTER.md`, `DEV_LOG.md` |
| `data/results/` | the final results reported in the dissertation |

---

## 2. Prerequisites

**Software**

| What | Version / note |
|---|---|
| Operating system | Windows 10/11, Linux or macOS for local runs; Linux (Rocky, Slurm) on ISCA |
| Python | 3.10 or later (tested on 3.10 and 3.11), with `pip` and `venv` |
| Git | any recent version |
| Ollama | a current release that serves the models below (https://ollama.com); `isca/setup.sh` installs it on ISCA |
| Python packages | installed by `pip install -e ".[dev,embed,ui]"` from `pyproject.toml`: requests, PyYAML, jsonschema, NumPy, scikit-learn, SciPy, radon, black; dev: pytest, statsmodels, pandas, python-docx, openpyxl; embed: PyTorch, transformers; ui: Streamlit |
| UniXcoder | `microsoft/unixcoder-base` from Hugging Face, downloaded on first use (needs `embed`) |
| matplotlib | for `scripts/frontier.py --plot` (`pip install matplotlib`) |
| Docker (optional) | for the isolated sandbox (`--sandbox docker`); the default is a subprocess sandbox with resource limits |

**Hardware**

| Model | Memory needed (4-bit) | Used on |
|---|---|---|
| Qwen3.5-9B, Llama 3.1 8B, Granite 4.2 8B, Gemma 3 12B | about 6 to 10 GB GPU memory | one A100 or an A100 MIG slice (10 GB); a laptop GPU works but is slow |
| Qwen3.5-27B | about 17 GB | one A100 80 GB |
| Gemma 4 31B | about 20 GB | one A100 80 GB |
| gpt-oss-120b | about 65 GB | one A100 80 GB |
| Qwen3.5-122B-A10B | about 70 GB | two A100 80 GB |

Disk: about 200 GB for all model weights, about 500 MB for traces, references and cache. Memory:
16 GB RAM for local runs; the ISCA jobs request 64 to 128 GB. The analysis needs no GPU.

**Accounts and access**

- For the reported runs: a University of Exeter ISCA account on project `COMM514`, and the
  University VPN when off campus.
- Internet access to download the models (Ollama library) and UniXcoder (Hugging Face).
- For the human rating study: a Microsoft 365 account for Microsoft Forms.

## 3. Setup


    python -m venv .venv
    .venv\Scripts\activate                 (Windows; on Linux/Mac: source .venv/bin/activate)
    pip install -e ".[dev,embed,ui]"
    copy config\models.example.yaml config\models.yaml      (Linux/Mac: cp)
    pytest                                 (all tests should pass)

`config/models.yaml` lists the model *profiles* (endpoint, model name, revision). Every LLM call
goes through the same OpenAI-compatible client and is cached by request hash in `data/cache/`.
Ollama must run with a context of at least 8192 tokens (checked before every run and stamped in
each trace).

**On ISCA** (all reported runs): follow `isca/README.md`. In short, inside an interactive job:
`bash isca/setup.sh` installs Python and Ollama under `~/macs-tools`, downloads the models, writes
`config/models.yaml` from `isca/models.isca.yaml` with pinned revisions, and runs the tests.

Model profiles used in the dissertation:

| Profile | Model | Sessions analysed |
|---|---|---|
| `qwen35_27b` | Qwen3.5-27B (main model) | 1,782 |
| `local` | Qwen3.5-9B | 1,782 |
| `qwen35_122b` | Qwen3.5-122B-A10B | 1,782 |
| `gpt_oss_120b` | gpt-oss-120b | 1,435 |
| `gemma4_31b` | Gemma 4 31B | 1,318 |
| `gemma3_12b` | Gemma 3 12B | 1,440 |
| `llama31_8b` | Llama 3.1 8B | 1,782 |
| `granite42_8b` | Granite 4.2 8B | 1,588 |
| | **total** | **12,909** |
| `deepseek` | DeepSeek-Coder-V2 Lite (AI judge only, not studied) | - |

Full design: all 27 tasks x all 11 conditions x seeds 1-6 (1,782 sessions). Qwen3.5-27B and 9B
ran the full design first; the other models started with a reduced design (24 main tasks;
conditions A, B, C, D, D_minus, E, E_minus; seeds 1-3) and were then extended towards the full
design with `isca/submit_split.sh`. Every session finished by the cut-off is analysed; the counts
above come from `docs/COMPUTE_TIME.md`.

---

## 4. Reproducing each part

Every step can be stopped and started again; it continues where it stopped.

### 4.1 Reference sets (needed before novelty can be scored)

100 ordinary solutions per task, per model, from a neutral prompt:

    python scripts/build_reference.py --tasks all --n 100 --profile qwen35_27b
    python scripts/tau_diagnostics.py --representation unixcoder      # Facilitator firing check

Output: `data/reference/models/<model>/<task>.jsonl`. On ISCA, `isca/run_model.sbatch` builds the
references itself before the sessions.

### 4.2 Sessions (the traces)

    python scripts/run_experiment.py --tasks all --conditions all --seeds 1-6 --profile qwen35_27b --dry-run
    python scripts/run_experiment.py --tasks all --conditions all --seeds 1-6 --profile qwen35_27b

Output: one trace per session in `data/traces/`. On ISCA (what was actually run):

    sbatch isca/run_model.sbatch                         # Qwen3.5-27B, full design
    PROFILE=local sbatch isca/run_model.sbatch           # Qwen3.5-9B, full design
    D="TASKS=main CONDITIONS=A,B,C,D,D_minus,E,E_minus SEEDS=1-3"
    env $D PROFILE=gpt_oss_120b sbatch isca/run_model.sbatch
    env $D PROFILE=gemma4_31b   sbatch isca/run_model.sbatch
    env $D PROFILE=qwen35_122b  sbatch --gres=gpu:2 -c 24 --mem=128G isca/run_model.sbatch
    bash isca/submit_split.sh llama31_8b mig             # small models: four jobs on task groups
    bash isca/submit_split.sh gemma3_12b gpu
    WORKERS=1 bash isca/submit_split.sh granite42_8b mig

Each job builds the references, runs the sessions, queues its own follow-up when the 24-hour limit
is reached, and finishes with the open-problem search. Progress: `bash isca/status.sh`.

**Cut-off.** The reported analysis uses the sessions finished by the cut-off recorded in
`docs/PREREGISTRATION.md` (by file time). Later traces are moved to `data/archive/` and not
analysed (`scripts/archive_traces.py` moves traces out of the analysed folder; nothing is deleted).

### 4.3 Analysis (results tables)

On ISCA, everything at once on a CPU node:

    sbatch isca/analyze_all.sbatch       # then download data/jobs/analysis_results.tgz

or step by step:

    python scripts/analyze.py --model qwen3.5:27b --out data/results/models/qwen3.5_27b/summary.json
    python scripts/analyze.py --model qwen3.5:27b --representation unixcoder --out data/results/summary_unixcoder.json
    python scripts/stats_report.py --tasks 24 --seeds 6                       # contrasts, CIs, Holm
    python scripts/stats_report.py --task-set contamination_check --out data/results/stats_contamination.json
    python scripts/model_consistency.py                                       # meta-analysis across models
    python scripts/frontier.py --plot                                         # quality-diversity frontier
    python scripts/mutation_test.py --model qwen3.5:27b                       # strength of the held-out tests

Output: `data/results/` (main model at the top level, every model in `data/results/models/`).

### 4.4 Extra studies

    python scripts/search_open.py run --tasks t3_circle_packing,t3_bin_packing,t3_cap_set --budget 40 --seeds 1-3
    python scripts/search_open.py report          # open problems: random vs single vs studio, equal budget
    sbatch isca/refsens.sbatch                    # reference-size sensitivity (50 / 100 / 200 references)
    python scripts/reference_sensitivity.py

### 4.5 Checking the families against labels (SC3)

48 solutions (8 from each of 6 tasks, chosen by rule) are labelled with their approach by three AI
raters through chat apps, each item in two option orders; a label counts only when both orders
agree.

    python scripts/rating_study.py build                       # 60 pairs and the 48 labelling items
    python scripts/ai_label_sheets.py export                   # sheets for the AI raters (two orders)
    python scripts/ai_label_sheets.py import --rater fable --order1 fable_order1.txt --order2 fable_order2.txt
    python scripts/calibrate_clustering.py                     # ARI sweep against the majority labels

The majority labels go to `data/calibration/labels.jsonl`; the frozen threshold and the sweep
table are recorded in `docs/PREREGISTRATION.md`.

### 4.6 Human validation of the novelty measure

Human raters compare 60 pairs of correct solutions ("more original?", "more creative?") in a
Microsoft Form while reading the booklet:

    python scripts/rating_study.py export                      # booklet (HTML) and Forms import file
    python scripts/rating_study.py import responses.csv        # the Form's export
    python scripts/rating_study.py judge                       # DeepSeek-Coder-V2 Lite AI judge (needs Ollama)
    python scripts/rating_study.py analyse                     # Bradley-Terry, Spearman, kappa

Output: `data/rating/results.json`. Individual responses are personal data under the ethics
approval and are not included; only the aggregate results are.

---

## 5. Compute time

The exact compute time of every run is in `docs/COMPUTE_TIME.md` (and
`data/results/compute_time.json`), produced from the traces and from Slurm's accounting records:

    # on ISCA, after the last job: every job's GPUs and elapsed time
    sacct -u $USER -S 2026-10-20 -E now -X -P \
      --format=JobID,JobName%40,Partition,AllocTRES%80,State,Start,End,Elapsed > data/jobs/sacct_all.txt
    # then, with the traces, job logs and that file in place
    python scripts/compute_time.py --sacct data/jobs/sacct_all.txt

It reports, per model: sessions, median session time, summed session time and the time the model
was busy (from the traces); and per model profile: GPU-hours on full A100 80 GB GPUs and on A100
MIG slices separately, and job wall hours (from Slurm, every job including failed and restarted
ones). Hardware: NVIDIA A100 80 GB on ISCA, up to six sessions at once against one Ollama server;
Qwen3.5-122B on two A100s; the small models partly on A100 MIG 1g.10gb slices.

## 6. Getting the same traces and references again

There are two ways, and they give different things:

1. **The exact data used in the dissertation.** The submission package contains every trace
   (`data/traces/`, and `data/archive/` for traces after the cut-off and earlier pilots), every
   reference set (`data/reference/`) and the response cache (`data/cache/`). Run section 4.3 on
   them: this reproduces every number in `data/results/` exactly.
2. **New runs.** Running sections 4.1 and 4.2 again generates new traces and references. Model
   outputs are sampled, so individual sessions will differ, but the design is identical and the
   results should agree within the reported confidence intervals. With the response cache from
   the package in `data/cache/`, identical requests are answered from the cache, so the earlier
   runs are replayed exactly.

Order for a complete rerun: setup (sections 2 and 3) -> references (4.1) -> sessions (4.2) -> cut-off ->
analysis (4.3) -> extra studies (4.4) -> labels and SC3 (4.5) -> human validation (4.6).

---

## 7. The submission package

Built with `python scripts/package_submission.py --tag dissertation-submission-v1.0 --data-tgz
macs_data_final.tgz` (the `data/` archive copied from ISCA). It holds the
code at the tag and all data: traces, archived traces, reference sets, response cache, results,
SC3 labels and the AI raters' answers, job logs with the Slurm accounting file, and the rating
study without personal data (responses under rater codes only). `config/models.yaml` is never
included. Next to the zip: `.sha256` (checksum of the zip) and `_MANIFEST.txt` (every file with
its size and SHA-256), so any change to the package can be detected.

## 8. What is in `data/results/`

| File | What |
|---|---|
| `summary.json`, `stats.json` | main model: per-condition measures and planned contrasts |
| `models/<model>/` | the same for every model |
| `model_consistency.json` | random-effects meta-analysis across the 8 models |
| `summary_unixcoder.json` | main model in UniXcoder space |
| `stats_contamination.json` | contamination-check tasks |
| `frontier.json`, `frontier.png` | quality-diversity frontier |
| `mutation.json` | mutation test of the held-out tests |
| `reference_sensitivity_qwen3.5_27b.json` | reference-size sensitivity |
| `search_report.json` | open-problem search |
| `tau_diagnostics*.json` | Facilitator threshold diagnostics |
| `compute_time.json` | sessions, busy time and GPU-hours per model (see section 5) |
