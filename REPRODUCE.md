# MACS: how it works and how to reproduce every result

This is the code as submitted with the MSc dissertation (COMM514, University of Exeter,
December 2026), frozen at git tag `dissertation-submission-v1.0`. Later work for publication is
done after this tag and does not change it. To get exactly this version from the repository:

    git checkout dissertation-submission-v1.0

Author: Arush Kumar Vishwakarma. Supervisor: Prof. Solomon S. Oyelere. Code MIT, data CC-BY 4.0.

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

## 2. Setup

Python 3.10 or later and [Ollama](https://ollama.com).

    python -m venv .venv
    .venv\Scripts\activate                 (Windows; on Linux/Mac: source .venv/bin/activate)
    pip install -e ".[dev,embed,ui]"
    copy config\models.example.yaml config\models.yaml      (Linux/Mac: cp)
    pytest                                 (135 tests should pass)

`config/models.yaml` lists the model *profiles* (endpoint, model name, revision). Every LLM call
goes through the same OpenAI-compatible client and is cached by request hash in `data/cache/`.
Ollama must run with a context of at least 8192 tokens (checked before every run and stamped in
each trace).

**On ISCA** (all reported runs): follow `isca/README.md`. In short, inside an interactive job:
`bash isca/setup.sh` installs Python and Ollama under `~/macs-tools`, downloads the models, writes
`config/models.yaml` from `isca/models.isca.yaml` with pinned revisions, and runs the tests.

Model profiles used in the dissertation:

| Profile | Model | Design |
|---|---|---|
| `qwen35_27b` | Qwen3.5-27B (main model) | full |
| `local` | Qwen3.5-9B | full |
| `qwen35_122b` | Qwen3.5-122B-A10B | reduced |
| `gpt_oss_120b` | gpt-oss-120b | reduced |
| `gemma4_31b` | Gemma 4 31B | reduced |
| `gemma3_12b` | Gemma 3 12B | reduced |
| `llama31_8b` | Llama 3.1 8B | reduced |
| `granite42_8b` | Granite 4.2 8B | reduced |
| `deepseek` | DeepSeek-Coder-V2 Lite (AI judge only, not studied) | - |

Full design: all 27 tasks x all 11 conditions x seeds 1-6. Reduced design: 24 main tasks x
conditions A, B, C, D, D_minus, E, E_minus x seeds 1-3.

---

## 3. Reproducing each part

Every step can be stopped and started again; it continues where it stopped.

### 3.1 Reference sets (needed before novelty can be scored)

100 ordinary solutions per task, per model, from a neutral prompt:

    python scripts/build_reference.py --tasks all --n 100 --profile qwen35_27b
    python scripts/tau_diagnostics.py --representation unixcoder      # Facilitator firing check

Output: `data/reference/models/<model>/<task>.jsonl`. On ISCA, `isca/run_model.sbatch` builds the
references itself before the sessions.

### 3.2 Sessions (the traces)

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

**Cut-off.** The reported analysis uses the sessions finished by 2 November 2026 12:00 (by file
time): 12,909 sessions over 8 models. Later traces were moved to `data/archive/` and are not
analysed (`scripts/archive_traces.py` moves traces out of the analysed folder; nothing is deleted).

### 3.3 Analysis (results tables)

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

### 3.4 Extra studies

    python scripts/search_open.py run --tasks t3_circle_packing,t3_bin_packing,t3_cap_set --budget 40 --seeds 1-3
    python scripts/search_open.py report          # open problems: random vs single vs studio, equal budget
    sbatch isca/refsens.sbatch                    # reference-size sensitivity (50 / 100 / 200 references)
    python scripts/reference_sensitivity.py

### 3.5 Checking the families against labels (SC3)

48 solutions (8 from each of 6 tasks, chosen by rule) are labelled with their approach by three AI
raters through chat apps, each item in two option orders; a label counts only when both orders
agree.

    python scripts/rating_study.py build                       # 60 pairs and the 48 labelling items
    python scripts/ai_label_sheets.py export                   # sheets for the AI raters (two orders)
    python scripts/ai_label_sheets.py import --rater fable --order1 fable_order1.txt --order2 fable_order2.txt
    python scripts/calibrate_clustering.py                     # ARI sweep against the majority labels

The majority labels go to `data/calibration/labels.jsonl`. Result: no threshold reached ARI 0.60,
so the pre-registered 3.0 was kept (PREREGISTRATION.md, 15 Nov 2026).

### 3.6 Human validation of the novelty measure

Human raters compare 60 pairs of correct solutions ("more original?", "more creative?") in a
Microsoft Form while reading the booklet:

    python scripts/rating_study.py export                      # booklet (HTML) and Forms import file
    python scripts/rating_study.py import responses.csv        # the Form's export
    python scripts/rating_study.py judge                       # DeepSeek-Coder-V2 Lite AI judge (needs Ollama)
    python scripts/rating_study.py analyse                     # Bradley-Terry, Spearman, kappa

Output: `data/rating/results.json`. Individual responses are personal data under the ethics
approval and are not included; only the aggregate results are.

---

## 4. Getting the same traces and references again

There are two ways, and they give different things:

1. **The exact data used in the dissertation.** All traces (12,909 analysed sessions plus the
   archived ones) and all reference sets are deposited as a data archive (Zenodo DOI given in the
   dissertation). Unpack them into `data/traces/` and `data/reference/models/`, then run
   section 3.3. This reproduces every number in `data/results/` exactly.
2. **New runs.** Running sections 3.1 and 3.2 again generates new traces and references. Model
   outputs are sampled, so individual sessions will differ, but the design is identical and the
   results should agree within the reported confidence intervals. To replay earlier runs exactly,
   copy the response cache (`data/cache/`) from the machine that made them: identical requests are
   then answered from the cache.

Order for a complete rerun: setup (section 2) -> references (3.1) -> sessions (3.2) -> cut-off ->
analysis (3.3) -> extra studies (3.4) -> labels and SC3 (3.5) -> human validation (3.6).

---

## 5. What is in `data/results/`

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
