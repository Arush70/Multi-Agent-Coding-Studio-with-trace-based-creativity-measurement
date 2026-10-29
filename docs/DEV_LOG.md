# Development log

Running record of changes, fixes and open decisions, kept so the weekly logs and the
dissertation's methods chapter can be written from it. Newest entries first. Each entry says
what changed, why, and which files. Pre-registration changes are also logged, dated, in
`docs/PREREGISTRATION.md` section 6.

---

## 2026-10-23 (Week 5) - six learners, 8192-token context, 2-learner run archived

- First 9B results (2 learners, 24 tasks x 6 seeds): studio beats tutor by about 2 families
  (Holm p < 0.001), beats equal-budget sampling, matches a single diversity prompt. Kept as the
  pilot in `data/archive/2026-10-23_two_learner_ctx4096/`.
- Learner factor is now 1 vs 6 (C, D, D_minus, E, E_minus, S, P), with C4 and D4 at 4 learners
  for a learner curve. Learners after the second see "the other learners" (plural). Secondary
  statistics: D4 - C4 and the change per extra learner (`learner_slope_fit`).
- Found that Ollama's 4096-token context cut some Critic prompts short (up to 4259 prompt tokens).
  `ModelProfile.context_length` (default 8192) is stamped into each trace and the cache key, and
  `run_experiment.py` asks the server (Ollama `/api/ps`) and refuses to run if it is lower. ISCA
  jobs run with 16384.
- `scripts/archive_traces.py` moves traces older than `COMPATIBLE_SINCE` (now 0.4.0) and their
  results into `data/archive/<name>/`. Trace schema 0.3.0 (C4, D4, learner_3..6). 129 tests pass.
- The small panel (Llama 3.1 8B, Gemma 3 12B, Granite 4.2 8B) moved from the laptop to ISCA:
  about a day of laptop time each, a few hours on an A100. Jobs now download a missing model
  themselves and pin every revision (`isca/pin_revisions.py`). Same design and settings as before.
- Six ISCA jobs (122B, Llama 4, Gemma 4, Llama 3.1, Gemma 3, Granite) started with Ollama on the
  CPU (`library=cpu`, GPU idle) and timed out; 27B, 9B and gpt-oss found the GPU. Likely cause:
  Ollama's GPU discovery timing out while loading its CUDA libraries from shared storage when
  several jobs started together. Jobs now copy Ollama to node-local disk, check the server log
  for `library=CUDA`, restart it up to three times, and stop rather than run on the CPU. The six
  jobs were cancelled; their references (partly built on the CPU) are archived and rebuilt.
- Root cause from the Ollama log: "llama-server GPU discovery watchdog timed out" (30 s). The A100s
  run with persistence mode off, so waking an idle GPU can take longer than that. Jobs now start a
  small torch helper that initialises CUDA and holds a context for the job's lifetime before
  Ollama starts, and Vulkan discovery is off.
- Small panel models ran at about 7 sessions an hour with the GPU 28% busy. Cause: each task's 100
  references were made one call at a time while the other workers for that task waited (about
  3 of every 4 hours). `build_reference.py --parallel N` now makes N samples at once (same seeds,
  same samples; order in the file may differ), the jobs use it with N = workers, and each job
  asks for 12 CPUs (24 for two-GPU jobs) with OMP threads capped at 2 per process.
- The GPU helper did not remove the discovery time-outs (about half of starts still failed). Jobs
  now try only the CUDA 12 library (`OLLAMA_LLM_LIBRARY=cuda_v12`, one 30 s try instead of three)
  and up to 10 times. Jobs are named `m-<profile>`; `isca/status.sh` prints jobs and sessions per
  model.
- Bug: open-problem search logs were named by task, strategy and seed but not model, so on ISCA
  the 27B search would resume the 9B's finished logs and skip itself. Logs now live in
  `data/search/<model>/`, a resume refuses another model's log, `search_open.py migrate` moves
  older flat logs, and the report covers every model (`main_model` = 27B). The search runs on
  27B and 9B only (pre-registered on the main model). The laptop's 9B search at 4096 context is
  archived with the 2-learner pilot.
- First full analysis (ISCA CPU job `isca/analyze_all.sbatch`). Mean novelty z came out near 1e10
  for 27B: on t1_mbpp19 its 100 references are near-identical, so the reference kNN spread is 0 and
  z has no scale (Llama 3.1 has two such tasks). `ReferenceStats.degenerate` (spread < 1e-6) now
  makes z undefined (None) for that task; the percentile-based correct-and-novel rate, the
  primary novelty outcome, is unaffected (anything unlike the repeated reference counts as rare).
- Rater form: Part A (60 pairs, SC4) and Part B (48 labelling items, SC3, `macs/rating/labelling.py`)
  in one Microsoft Form with an information text and a consent question; raters are classmates and
  acquaintances, never the researcher or supervisor. `rating_study.py` builds, exports, imports and
  analyses both parts; Part B majority labels go to `data/calibration/labels.jsonl` for the ARI
  sweep. The AI rater (`judge`) now also labels Part B (two option orders, must agree), written to
  `data/calibration/judge/`. `isca/rating.sbatch` builds the materials and runs the AI rater on ISCA.
- 2026-10-28: eight of nine models complete (27B, 9B full design; 122B, gpt-oss-120b, Gemma 4,
  Llama 3.1, Gemma 3, Granite reduced design). Llama 4 Scout never started: Ollama's GPU discovery
  timed out on every two-GPU start (gpu11, gpu13), and three earlier Llama 4 jobs had received
  `CONDITIONS=A` only because `sbatch --export=...,CONDITIONS=A,B,...` splits on commas (fixed by
  passing variables through the environment). Llama 4 is retried on one A100 with
  `CONTEXT=8192` and 3 workers (`OLLAMA_CONTEXT_LENGTH` is now set from `CONTEXT`).
- 2026-10-29: the A100 partition was full for hours while V100 (`volta`) and A100 MIG slices
  (`gpu-mig`, 10 GB) were idle. Models that fit were moved there (Llama 3.1 8B and Granite 4.2 8B on
  MIG slices; Gemma 3 12B, Gemma 4 31B and the 27B reference-sensitivity job on V100s, context 8192).
  Follow-up jobs now keep their partition and GPU type (`GRES`). Consequence for the write-up: within
  one model, the sessions added for the identical design ran on different GPU hardware from the
  first 504 (same model file and quantisation; floating-point differences between GPUs can change
  sampled tokens, which adds noise but not a systematic effect). Each job's node and partition are
  in its log (`data/jobs/isca-<job>.log`).

## 2026-10-22 (Week 5) - University GPU (ISCA): 27B becomes the main model

- ISCA access arrived (A100 80 GB, account COMM514). By the 15 Oct decision rule the main model
  is now Qwen3.5-27B; the laptop's full 9B run continues and is reported as a second model.
- `isca/setup.sh` installs Miniforge, Ollama, the models and UniXcoder in `~/macs-tools` and
  writes `config/models.yaml` with revisions pinned from `ollama list`.
- `isca/run_model.sbatch` runs one model on a full A100: per task it builds the 100 references,
  then the sessions, 6 at a time against one Ollama server (context 4096, as on the laptop). It
  resumes after the 24 h limit through a queued follow-up job, then runs the open-problem search
  and packs the results for download.
- Panel extended with large open models: Qwen3.5-122B-A10B, gpt-oss-120b, Llama 4 Scout,
  Gemma 4 31B (reduced design). Recorded in PREREGISTRATION before any main-run analysis.

## 2026-10-20 (Week 5) - extended design: new conditions, models, tasks, rating study, open problems

Everything below is pre-registered (PREREGISTRATION, 2026-10-20) before any session in the new
conditions was run. Orchestrator 0.3.0 (0.2.3 traces stay comparable; the core conditions did not
change), role prompts 0.3.0, trace schema 0.2.0.

| Change | Why | Files |
|---|---|---|
| Conditions defined in one registry; E, E_minus (neutral Innovator, Facilitator on/off) | The Facilitator rarely fired because the Innovator is told to differ; E/E_minus give it convergence to act on (RQ2) | `macs/conditions.py`, `macs/orchestrator.py`, `macs/prompts/roles.yaml` |
| Baselines S (tutor sampled to the studio's token budget) and P (one prompt for K different approaches) | Rules out "the studio just makes more tries" and "one diversity prompt does the same" | `macs/orchestrator.py` (`_run_sampling`, `_run_single_prompt`, `split_approaches`) |
| Statistics: pairwise and studio-2x2 contrasts, interactions, Holm family of 10, MDE at alpha/10 | The new questions need planned tests | `macs/analysis/stats.py`, `scripts/stats_report.py` |
| Several models: per-model reference sets (`data/reference/models/<model>/`), runner takes several profiles, per-model analysis, cross-model random-effects meta-analysis | Is the result the same on other models? | `macs/metrics/reference.py`, `scripts/run_experiment.py`, `scripts/analyze.py --all-models`, `scripts/model_consistency.py`, `config/models.example.yaml` |
| Tasks: HumanEval/20, 26, 57 (24 main tasks, SC1); EvoEval_creative/13, 16, 31 as a contamination-check set; circle packing (n = 26) as an open problem | SC1 asks for 24; memorisation check; a problem with a published best-known value | `tasks/tier2/*`, `tasks/tier3/t3_circle_packing`, `macs/tasks/bank.py` (`task_set`), `docs/TASK_SOURCES.md` |
| Learner contribution (RQ3): uptake of learner-mentioned constructs, learner-led share, rarity of learner ideas | RQ3 had no measure of the learners' influence | `macs/metrics/learners.py`, `macs/analysis/aggregate.py` |
| Rating study tooling: 60 blind pairs, booklet + Microsoft Forms Quick-import document, response import, Bradley-Terry, kappas, Spearman with bootstrap CI; AI judge asked in both orders | SC4 by 9 Nov | `macs/rating/*`, `scripts/rating_study.py` |
| Open-problem search: random / single / studio at an equal budget of scored programs, logs, best-so-far curves, report | Can the studio find better solutions to hard and open problems? | `macs/search/loop.py`, `macs/prompts/search.yaml`, `scripts/search_open.py` |
| UniXcoder as the parallel representation (families and novelty), calibration in that space | Proposal: both representations, analysed separately | `macs/analysis/aggregate.py`, `scripts/analyze.py --representation`, `scripts/calibrate_clustering.py --representation` |
| Mutation testing (AST mutator), quality-diversity frontier and Tier 3 payoff from per-proposal records | Test strength; does creativity pay off | `macs/analysis/mutation.py`, `scripts/mutation_test.py`, `scripts/frontier.py` |
| Blind labelling: code shown without comments/docstrings; no pre-selected answer | The model's comments named the strategy and could unblind SC3 | `macs/ui/pipeline.py` (`blind_view`), `macs/ui/app.py` |
| Scoring is the same on every Python version: the sandbox runs `from typing import *` before the candidate's code | Found by re-scoring the pilot under Python 3.10: 9 HumanEval/0 studio sessions lost correct proposals because `List` was undefined (Python up to 3.13 evaluates annotations eagerly; the laptop's 3.14 does not). HumanEval's own evaluation runs prompt + completion, so the import is in scope there too | `macs/sandbox/runner.py`, `tests/test_sandbox.py` |
| Batch runner pauses after 3 failures in a row and stops after 6 (model server down) | A long run should not mark every remaining session as failed | `scripts/run_experiment.py` |
| Dashboard: Rating study and Open problems tabs; models, frontier and mutation in Statistics; reference sensitivity; plain-language conclusion; UniXcoder view; example outputs in Traces; several models in Multi-run | Everything stays runnable from the browser | `macs/ui/app.py`, `macs/ui/extra_tabs.py` |

Tests: 124 passing (was 91). The dashboard renders all ten tabs headless (Streamlit AppTest).

## 2026-10-16 (Week 4) - the whole pipeline in the dashboard

Everything that needed the terminal can now be done from `streamlit run macs/ui/app.py`:

| Tab | What it does | Runs |
|---|---|---|
| Sidebar | model profile (from config/models.yaml, keys hidden) and sandbox, used by every tab; job status | - |
| Overview | "Analyse all traces" (writes summary.json), Studio vs Tutor, per-condition table | `analyze.py` |
| Statistics | S8 contrasts with 95% CIs, Holm, MDE for a chosen design (tasks, seeds) | `stats_report.py` |
| References | per-task status (ready / incomplete / missing / other model); clear to `data/reference_archive/` (never deleted); generate (missing only, all, or chosen, samples per task); Facilitator firing check | `build_reference.py`, `tau_diagnostics.py` |
| Multi-run | tasks (all, tier, chosen), conditions, **number of seeds** (default 6, as in the proposal) and first seed; shows sessions total / done / to run for this model, warns about tasks without a ready reference and a missing torch; progress bar and Stop; single quick session | `run_experiment.py`, `run_session.py` |
| Labelling (SC3) | blind labelling pool (condition, strategy and automatic family hidden; fixed shuffled order), one solution at a time with the task's known families plus `other` / `unsure`; two raters with Cohen's kappa; calibration sweep | `calibrate_clustering.py` |
| Setup | model server reachable, model available, torch/transformers, Docker, statsmodels | - |

Long steps run as background jobs (`macs/ui/jobs.py`): the script runs in its own process through a
small wrapper that records the exit code, output goes to `data/jobs/<kind>.log`, and the page
reads progress from it. Refreshing or closing the page does not stop a job; Stop ends the job and
everything it started. Two jobs that use the model cannot run at the same time.

Fixes found while building it:
- `run_experiment.py` counted a cell as done whatever model produced it; it now counts only
  sessions from the chosen profile's model, so switching models re-runs them.
- A long comma-separated task list was tried as a directory name and crashed with "File name too
  long" (`build_reference.py` / `run_experiment.py`); it is now only treated as a path when it can be.
- `calibrate_clustering.py` ignores `unsure` labels.

The layout was exercised end to end against a stand-in Streamlit with the pilot data (every tab,
every button); background jobs, reference status/archiving, run plans and labelling are covered by
`tests/test_ui_pipeline.py`. Files: `macs/ui/app.py`, `macs/ui/jobs.py`, `macs/ui/pipeline.py`,
`scripts/run_experiment.py`, `scripts/build_reference.py`, `scripts/calibrate_clustering.py`,
`pyproject.toml` (`ui` extra), `README.md`, `.gitignore` (`data/jobs/`).

---

## 2026-10-15 (Week 4, S8) - statistics: mixed models, planned contrasts, Holm, MDE

`macs/analysis/stats.py` fits the proposal's model (3.8) for each outcome:
`outcome ~ studio * two_learners + (1 | task)` by REML, and `outcome ~ facilitator + (1 | task)` for
D vs D_minus. With one random intercept the REML likelihood is profiled over one variance ratio,
so only numpy and scipy are needed. Checked by simulation (estimates unbiased; standard errors
match the empirical spread; 95% intervals cover the true value 94% of the time over 800 runs) and,
where statsmodels is installed, against `statsmodels` MixedLM in `tests/test_stats.py`.
`scripts/stats_report.py` prints the planned contrasts with 95% CIs, Holm-adjusted p-values for
the five primary contrasts, the secondary outcomes, and the MDE for a chosen design. The
specification is recorded in PREREGISTRATION (2026-10-15).

On the 0.2.3 pilot (3 tasks, planning only): residual SD of the correct-and-novel rate 0.30 and of
matched families 0.40. For 21 tasks x 6 seeds (Holm-strict alpha 0.01, 80% power) the MDE is about
0.13 for the correct-and-novel rate and 0.42 families for B - A / D - C; the pilot differences
were 0.28-0.47 and 0.83-1.03 respectively, so the planned design is large enough for effects of
the pilot's size. For D - D_minus the pilot difference was -0.08 families with the Facilitator
firing in 2 of 18 sessions, consistent with the RQ2 limitation.

Files: `macs/analysis/stats.py`, `scripts/stats_report.py`, `tests/test_stats.py`,
`pyproject.toml` (scipy declared; statsmodels and pandas in `dev` for the cross-check), `README.md`.

---

## 2026-10-15 (Week 4, results) - pilot under orchestrator 0.2.3

45 of 45 sessions completed (3 tasks x 5 conditions x 3 seeds, `qwen3.5:9b`, thinking off,
subprocess sandbox, so development numbers only). Pilot figures describe the procedure; they are
not results of the study and carry no inference (3 tasks, 3 seeds).

| Condition | Proposals correct | Families at matched m (m = 3) | Mean novelty z | Correct and novel | Tokens / session |
|---|---|---|---|---|---|
| A (Tutor, 1 learner) | 9/9 | 1.67 | -0.01 | 0.22 | 652 |
| B (Studio, 1 learner) | 44/55 (80%) | 2.69 | 1.79 | 0.50 | 11,816 |
| C (Tutor, 2 learners) | 9/9 | 1.67 | -0.16 | 0.11 | 1,105 |
| D (Studio, 2 learners) | 46/59 (78%) | 2.50 | 2.02 | 0.58 | 16,955 |
| D_minus (Facilitator off) | 46/58 (79%) | 2.58 | 1.98 | 0.59 | 16,667 |

Procedure checks:
- **Facilitator** (UniXcoder, p25): fired in 2 of 18 B/D sessions, both on t1_mbpp19; D and D_minus
  are identical sessions on the other two tasks. As anticipated in PREREGISTRATION (RQ2 limitation).
- **Selection**: Critic chose directly in 21 of 27 studio sessions, after the reminder in 4,
  fallback in 2 (was 6 of 26 before the reminder).
- **Code parse rate**: 162 of 172 studio artefacts parse (94%); all 18 tutor outputs parse.
- **Token ceiling**: no session stopped; the largest used 38,702 of 40,000 (t2_humaneval0, D,
  seed 3), so the ceiling is close to binding on harder tasks.
- **Tutor repeats itself across seeds**: on two tasks all three tutor seeds produced code with the
  same novelty score, so the tutor's pooled diversity at matched m comes mostly from one task.
- **Learners** (A vs C, B vs D): no visible difference at this size.

Next, per the plan: SC3 hand-labelling to freeze the clustering threshold; references for all 21
tasks; minimum detectable effect from pilot variance (needs the S8 statistics code).

---

## 2026-10-15 (Week 4, decision) - Facilitator measures in UniXcoder space (rule applied)

The pre-registered rule (PREREGISTRATION, 2026-10-15) was applied to the laptop diagnostics. No
space or percentile gave a 10-90% firing rate on all three pilot tasks; each option had exactly one
task in the band; the tie-break chose **UniXcoder at the 25th percentile**.

| Would-fire rate (D_minus traces) | p25 | p50 | p75 |
|---|---|---|---|
| AST, t1_mbpp140 | 0.00 | 0.17 | 0.83 |
| AST, t1_mbpp19 | 0.83 | 1.00 | 1.00 |
| AST, t2_humaneval0 | 0 | 0 | 0 |
| UniXcoder, t1_mbpp140 | 0 | 0 | 0 |
| UniXcoder, t1_mbpp19 | 0.17 | 0.33 | 0.33 |
| UniXcoder, t2_humaneval0 | 0 | 0 | 0 |

Reading: the studio's proposals are far more spread out than ordinary output (96th-100th
percentile in UniXcoder space), so "as converged as ordinary output" is rarely reached. This
supports RQ1 and limits RQ2; both are stated in PREREGISTRATION. Worth raising at supervision: a
stronger RQ2 test would need a condition in which the Innovator is not told to differ.

| Change | Files |
|---|---|
| `FACILITATOR_REPRESENTATION = "unixcoder"`; reference embedded once per process per task | `macs/orchestrator.py` |
| A missing torch/transformers install stops a studio run with an install message, instead of silently using the default tau | `macs/orchestrator.py`, `scripts/run_experiment.py`, `scripts/run_session.py` |
| Tests use a deterministic stand-in embedding (no torch or model download in CI) | `tests/conftest.py`, `tests/test_orchestrator.py` |
| Setup installs the `embed` extra; workflow lists the firing check | `README.md` |

Orchestrator 0.2.3. The pilot is re-run once more; tutor sessions come from the cache, studio
sessions change where the Facilitator now fires.

---

## 2026-10-15 (Week 4, pilot review) - selection reminder, code extraction, switchable Facilitator space

From the 43 completed pilot sessions (orchestrator 0.2.1):

- **Critic selection fell back in 6 of 26 studio sessions.** In every case the 9B, shown all five
  proposals with their code, wrote an implementation instead of a `SELECTED: pN` line. Change: if
  the reply has no valid selection, the Critic is sent one fixed reminder ("Reply with exactly one
  line, SELECTED: <id>, naming one of p0, ..., p4. Do not write code.", seed s+251); only if that
  also fails is the deterministic fallback used. The selection event records `critic`,
  `critic_after_reminder` or `fallback:...`, and its tokens include the reminder call. No prompt in
  `roles.yaml` changed.
- **12 of 164 studio code outputs did not parse.** Two were replies that opened a ```python block
  and never closed it, so the extractor kept the whole reply; code extraction now takes an unclosed
  block to the end of the reply and accepts the `py` tag. The other ten are model failures (prose
  instead of code, or code cut off) and count as incorrect proposals; the parse rate is a pilot
  statistic. All 18 tutor outputs parsed.
- **Facilitator space is switchable.** `FACILITATOR_REPRESENTATION` ("ast" or "unixcoder") sets the
  space for both the Facilitator's distances and tau; the trace header records it. It stays "ast"
  until the pilot decision rule (PREREGISTRATION, 2026-10-15) is applied to the UniXcoder numbers.
- `tau_diagnostics.py` skips sessions that crashed, and skips unparseable code in UniXcoder mode as
  in AST mode.

Orchestrator 0.2.2 (selection and extraction changed). Re-running the pilot is cheap: unchanged
requests are served from the response cache.

---

## 2026-10-15 (Week 4, pilot) - first 9B pilot, encoding fix, Facilitator firing check

**Pilot run** (orchestrator 0.2.1, `qwen3.5:9b`, 3 tasks x 5 conditions x 3 seeds = 45 sessions):
43 completed, 2 failed (t1_mbpp140 D and D_minus, seed 2). Every session stayed under the token
ceiling (largest 26,608). Tutor sessions used about 450-1,550 tokens, studio sessions about
6,900-26,600.

**Bug: text encoding on Windows (fixed).** The model wrote "α" in a strategy; the response cache was
written with the Windows code page, which cannot encode it, so the write failed half-way and left a
broken cache file. The next session (D_minus, which sends the same first requests as D) then failed
reading it. 22 earlier cache entries had also been written in the Windows code page. Fix: every
file the system reads or writes now states UTF-8 (cache, sandbox candidate and harness files, task
and prompt YAML, schema, results); cache entries are written to a temporary file and moved into
place, so a crash cannot leave a half-written entry; an unreadable entry counts as a cache miss and
is fetched again; entries in the old code page are still read. Prompts and task files are ASCII, so
no prompt hash or cache key changes. Files: `macs/llm/client.py`, `macs/sandbox/runner.py`,
`macs/trace/writer.py`, `macs/tasks/bank.py`, `macs/orchestrator.py`,
`macs/analysis/heuristic_score.py`, `macs/ui/app.py`, `scripts/*.py`.

**Finding: the Facilitator fires on one pilot task only.** Interventions in B and D sessions:
t1_mbpp19 7, t1_mbpp140 0, t2_humaneval0 0. Where it never fires, D and D_minus are the same
session (identical tokens), so the D vs D_minus contrast carries no information on those tasks.
Cause: the Innovator is told to differ from earlier strategies, so studio proposals are usually
more spread out than random reference solutions. The studio's window values sit at about the 20th
percentile of the reference windows on t1_mbpp19, the 33rd on t1_mbpp140 and the 98th on
t2_humaneval0 (AST features; container numbers, which differ slightly from the laptop because
`black` and `radon` were not installed there). A threshold at the reference's 25th percentile
therefore fires on mbpp19 and almost never elsewhere. The proposal requires tau to be checked on
pilot tasks for a non-degenerate firing rate; that check fails in AST space. The decision rule for
resolving it is fixed in `docs/PREREGISTRATION.md` (2026-10-15, before the UniXcoder numbers are
seen). `scripts/tau_diagnostics.py` now checks D_minus traces by default (Facilitator off, so its
own instructions do not shape the proposals), reports the percentile rank, and writes
`data/results/tau_diagnostics_<representation>.json`.

**Also noted for the pilot review:** the Critic's selection fell back to "most public tests"
in 6 of 26 studio sessions (its reply had no `SELECTED: pN` line).

---

## 2026-10-15 (Week 4, later) - tau rule matched to the Facilitator, no check after the last proposal

**Finding.** With the Qwen3.5-9B reference sets (100 per task, built 15 Oct) tau was still 0 on all
three pilot tasks, so every studio session would run on the uncalibrated default 0.05. The model is
not the cause: the 9B wrote 78-100 textually different programs per task, but only 11-19 distinct
AST-feature vectors, and 90-95% of reference solutions have an identical twin in that space. The
rule took the nearest of 99 other solutions, so the 25th percentile was always 0. The Facilitator,
by contrast, compares a proposal with only 2-4 earlier ones.

Options measured on the real references and the 9B test session (no model calls):

| Option | tau (mbpp19 / mbpp140 / humaneval0) | Outcome |
|---|---|---|
| Original rule, raw AST | 0 / 0 / 0 | degenerate, default used |
| Original rule on distinct shapes only | 0.003 / 0.007 / 0.002 | far below the studio's window values (0.03-0.05), almost never fires |
| Original rule, standardised AST | 0 / 0 / 0 | degenerate |
| **Matched rule, raw AST (adopted)** | **0.050 / 0.022 / 0.003** | non-zero on all tasks; fired at 1 of 2 checks on the 9B test session |

**Adopted.** tau = the 25th percentile of the Facilitator's own window statistic on 2,000 random
sequences of 5 reference solutions (fixed seed), in the same space (raw AST, cosine). It keeps the
proposal's percentile, frozen reference and space, and fixes the mismatch in the statistic.
Standardised AST was rejected for the Facilitator: centring on the reference mean makes all studio
proposals point the same way (they are all longer than typical), so they look close to each other.
UniXcoder (the proposal's other representation) is to be checked with the same script before the
main-run freeze; it needs torch, so it runs on the laptop.

| Change | Files |
|---|---|
| `tau_from_reference_sequences` (matched rule); `resolve_tau` uses it; header records `tau_rule` | `macs/facilitator.py`, `macs/orchestrator.py` |
| No convergence check after the final proposal (the 12 and 15 Oct traces each logged an intervention after p4 that could not affect anything) | `macs/orchestrator.py` |
| Orchestrator version 0.2.1 (session behaviour changed); 0.2.0 traces are excluded by default | `macs/orchestrator.py` |
| `scripts/tau_diagnostics.py`: both rules, twin share and firing rate per task; `--representation unixcoder` | `scripts/tau_diagnostics.py` |
| Tests for the matched rule and for no intervention after the last proposal | `tests/test_orchestrator.py`, `tests/test_scripts.py`, `tests/test_analysis.py` |

---

## 2026-10-15 (Week 4) - model switch to Qwen3.5-9B, thinking off, model guards

**Model.** Development and the pilot move from `qwen2.5-coder:7b-instruct-q4_K_M` (Ollama id
`dae161e27b0e`) to **`qwen3.5:9b`** (Ollama id `56671c2ab938`, Apache 2.0, about 6.6 GB, fits the
laptop GPU). Reason: the 7B gave near-identical reference solutions, so tau fell back to its
default and novelty saturated (risk already named in the project plan). The proposal (3.9) names
"the largest Apache-2.0 Qwen coder checkpoint" for the main run; Qwen3.5 is the newer Apache-2.0
Qwen generation and is a general model, not a coder line, so this is a recorded deviation.
The main-run model follows the decision rule in `docs/PREREGISTRATION.md` (2026-10-15).

**Thinking off.** Qwen3.5 in Ollama reasons before answering by default. Checked on the
OpenAI-compatible endpoint: with `reasoning_effort: none` the reply has no reasoning and used
342 completion tokens, against 689 with reasoning on, for the same request. The `local` profile
sets this through the new `extra_body` field.

| Change | Files |
|---|---|
| A model profile can carry `extra_body`, extra fields merged into every chat request. It is recorded in the trace header and is part of the cache key only when set, so existing caches keep their hashes. A leading `<think>...</think>` block in a reply is removed from the returned text; the full reply stays in the cache. | `macs/llm/client.py`, `config/models.example copy.yaml` |
| The analysis never pools traces from different models: by default it analyses the model of the most recent trace and reports the rest as skipped; `analyze.py --model NAME` chooses one. | `macs/analysis/aggregate.py`, `scripts/analyze.py` |
| `build_reference.py` refuses to top up a reference file that holds samples from a different model or revision (move it aside or use `--overwrite`), so a reference set always comes from one model. | `scripts/build_reference.py` |

The 7B traces from 12 October and the 30-sample 7B reference for `t1_mbpp19_test_duplicate` are
development data only; they were moved to `data/traces_archive/` and `data/reference_archive/`.

---

## 2026-10-09 (Week 3) - open decisions D1-D6 implemented, batch runner added

All six open decisions from 2026-10-07 are now implemented, following the proposal. Session
logic changed, so the orchestrator version is now **0.2.0**; it is written into every trace
header and the analysis skips older traces by default (`--include-old` keeps them). Role prompts
are now version 0.2.0 (one prompt added, none changed).

| # | What was done | Proposal reference | Files |
|---|---|---|---|
| D1 | **Reference sets of 100 per task.** `build_reference.py` now tops up existing files instead of skipping them (30 -> 100 appends seeds 30..99; same result as building 100 fresh because seeds are fixed), is safe to stop and restart, and accepts `tier3` and comma-separated task ids. The analysis always uses the **first 100 by seed**, so building 200 for the sensitivity check never changes the main result. New `scripts/reference_sensitivity.py` compares novelty at 50 / 100 / 200 (Spearman rank correlation and agreement of the "novel" flag). `analyze.py` warns when a reference has fewer than 100. Trace header records `reference_n`. | 3.6: "100 solutions ... sensitivity to 50 and 200 is checked" | `scripts/build_reference.py`, `macs/metrics/reference.py`, `scripts/reference_sensitivity.py`, `scripts/analyze.py` |
| D2 | **Up to 3 critique-revision rounds.** At least one round always runs; the loop stops early once a revision passes every public test, otherwise it stops after 3. Each revision builds on the previous one (`p_rev1 -> p_rev2 -> p_rev3`, linked by `parent_ids`). | 3.2: "bounded critique-revision (max 3 rounds)" | `macs/orchestrator.py` |
| D3 | **Facilitator can act earlier.** The window is now three *proposals* (as the proposal words it), not three distances (which needed four proposals). It can now fire after proposal 3 and shape proposals 4 and 5. K stays 5 and the window stays 3. | 3.3: "mean of these distances over a window of three proposals" | `macs/facilitator.py` (`window_metric`) |
| D4 | **Token ceiling enforced.** All model calls in a session go through a counter; when the next call would start at or above the ceiling (40,000 prompt + completion tokens), the session stops cleanly and `session_end` says `stopped: token ceiling ...`. Every finished session's `session_end` records `tokens_used`. The analysis now reports mean tokens per condition and how many sessions hit the ceiling. A normal studio session uses well under the ceiling, so it is a safety stop, not a constraint on normal runs. | 3.2: "All conditions share stopping rules and token ceilings, logged so compute can be reported"; 3.1: "token counts per cell are reported with every contrast" | `macs/orchestrator.py`, `macs/trace/writer.py` (`end_note`), `macs/analysis/aggregate.py` |
| D5 | **Specific Facilitator instructions.** When it fires, the Facilitator now names what the converging proposals share: the construct most of the last three implementations use (a set, a dictionary, sorting, recursion, a heap, bit manipulation, nested loops; read from the code's syntax tree) and the name of the latest approach (first sentence of the Innovator's strategy). Example: "Propose a strategy that does NOT use a set." Hybridisation now gets the two approach names instead of two whole paragraphs. The full instruction is written into the intervention event. | 3.3: "solve without a hash map", "all proposals so far sort the input" | `macs/facilitator.py` (`code_constructs`, `dominant_construct`, `approach_name`), `macs/orchestrator.py` |
| D6 | **The Critic selects a proposal.** After the five proposals, the Critic is shown all of them (strategy, code, public-test result) and replies `SELECTED: pN` with a reason. Critique and revision then work on that proposal. If the reply has no valid selection, the proposal passing the most public tests is taken (earliest on a tie), and the selection event says `fallback`. The clustering summary is still logged on the same event. New role prompt `critic_selection`. | 3.2: "selection ... implementation, bounded critique-revision"; "Critic ... may argue for a rejected family" | `macs/orchestrator.py`, `macs/prompts/roles.yaml` |

### Workflow added

- `scripts/run_experiment.py`: runs tasks x conditions x seeds in one command
  (`--tasks all --seeds 1-6`), `--dry-run` shows the plan, safe to stop and restart (a cell with a
  complete trace from the current orchestrator is skipped; a crashed session is re-run), one
  failing session does not stop the batch, prints time remaining.
- `scripts/analyze.py`: new columns `tokens` and `interv` (mean Facilitator interventions); skips
  older-version and crashed traces with a printed note; `--include-old` to keep them.
- Dashboard headline and family charts now use families at a matched number of solutions; the
  old "tutors 1, studios 6, the core prediction" text is removed.
- README has a Workflow section.

### Checks

- 62 tests pass (was 48). New tests cover each item above, the batch runner's resume logic, the
  reference top-up (including recovery from a half-written line) and the dashboard headline.
- End-to-end dry run with a fake model: 2 tasks (one Tier 1, one Tier 3) x 5 conditions x 2 seeds
  through `run_experiment.py`, then `analyze.py`, `calibrate_clustering.py --make-stub` and
  `reference_sensitivity.py`. All traces valid; re-running skipped all 20 done cells; the
  Facilitator fired only in B and D, never in D_minus.

### Still to do

- Build the 100-sample references on the real model and re-run the pilot (old traces are skipped).
- If references are still degenerate at 100 (the 7B model may keep producing the same 2 shapes),
  tau falls back to the default and novelty stays saturated; that is a model limitation to state,
  and a reason for the stronger model in the main run.
- The Facilitator's "optional return to divergence" (extra proposals after the last one) is not
  implemented: K stays fixed so every studio condition has the same proposal budget.

---

## 2026-10-07 (Week 3) - code review: four bugs fixed, pilot result withdrawn

### Context

A review of the code against the traces found that the conditions were not actually different
from each other on the pilot task. The design follows the supervisor's direction where it differs
from the proposal (see "Intentional deviations" below); only things that were genuinely broken were
changed.

### Fixed

| # | Problem found | Evidence | Fix | Files |
|---|---|---|---|---|
| F1 | **Learner turns never reached the solver.** Learners were logged, but neither the Tutor nor the Innovator was shown what they said. The solver got identical input in A and C, and in B and D, so the learner factor (RQ3) could not have any effect. | Pilot traces on `t1_mbpp19_test_duplicate`: A and C produced byte-identical code; B, D and D_minus produced byte-identical code (same cached responses). | Learner turns are now passed to the Tutor and the Innovator. In two-learner conditions, learner 2 sees learner 1's turn (they work together). | `macs/orchestrator.py` (`_run_learners`, `_learner_block`) |
| F2 | **Facilitator threshold tau was 0 on tasks with a reference set, so it could never fire.** Two causes: (a) tau was read off the reference's *standardised* features while the Facilitator measures cosine distance on *raw* features, so the two numbers were in different spaces; (b) the 30-sample reference has only 2 distinct solutions, so most nearest-neighbour distances are 0 and the 25th percentile is 0. | `tau_from_reference` on the mbpp19 reference returns 0.0; no D trace on that task contains an intervention. (On tasks with no reference the default tau = 0.05 was used, and the Facilitator did fire there: mbpp88, humaneval0, bin packing.) | tau is now computed in the Facilitator's own space. If it comes out as 0 (degenerate reference) the documented default is used. The proposal already requires tau to be checked for a non-degenerate firing rate, so this applies that check rather than inventing a new rule. Every studio trace header now records `tau` and `tau_source` (`reference`, `default:no_reference`, `default:degenerate_reference` or `override`). | `macs/orchestrator.py` (`resolve_tau`, header) |
| F3 | **Critic and revision were not shown the code.** The Critic received only the strategy text, although its prompt says "examine the implementation". The revision prompt said "revise your solution" without including any solution, so the Engineer was writing from scratch. | `_run_studio` prompt construction. | The Critic now gets the strategy plus the implementation; the revision prompt includes the current code plus the critique. | `macs/orchestrator.py` |
| F4 | **Trace header claimed 3 revision rounds while the code ran 1.** | `MAX_REVISIONS = 3` written to the header; loop runs once. | Constant set to 1 so the header is truthful. Whether to implement up to 3 rounds is an open decision (D2). | `macs/orchestrator.py` |
| F5 | **Family count logged in the trace disagreed with the analysis.** The "selection" event clustered raw features at threshold 0.30; the analysis clusters within-sample standardised features at 3.0. | Two `CLUSTER_THRESHOLD` constants in two files. | The selection event now uses the analysis method and threshold. One value, one place (`macs/analysis/aggregate.py`). | `macs/orchestrator.py` |
| F6 | **"Tutor 1 family vs studio 6" was true by construction.** A tutor session writes 1 program, so it can only ever be 1 family; a studio session writes 6, and within-session standardisation split all 6 into separate families (entropy 2.585 = log2 6, the maximum). | Pilot summary. | Added a fair comparison (kept the old columns): per task, all solutions from all conditions and seeds are clustered together in one common standardised space, and each condition's family count is read off its rarefaction curve at the **same** number of solutions m. Reported by `scripts/analyze.py` and saved as `pooled_diversity` in `summary.json`. The tutor conditions get more than one solution from repeated seeds, so this needs several seeds per cell. | `macs/analysis/aggregate.py` (`pooled_diversity`), `scripts/analyze.py` |

Tests added for F1, F2, F3, F4 and F6 (`tests/test_orchestrator.py`, `tests/test_analysis.py`).
Each new orchestrator test fails on the old code and passes on the new.

### Consequence: the Week 2 pilot result is withdrawn

The Week 2 headline ("tutors 1 family, studios 6") came from two sessions' worth of output
listed under five condition names (F1), compared at unequal sample sizes (F6), with the
Facilitator unable to fire on that task (F2). The existing traces in `data/traces/` were produced
by the old code and should not be used for results. Re-run the pilot with several seeds per cell.

Corrections to earlier logs, for the record:
- Week 2 log, "the Facilitator fired on convergence": true for tasks without a reference set
  (default tau), not for the pilot task with a reference set (tau = 0).
- Week 2 log, "41 tests passing": the suite had 48 test functions after these fixes (62 after 2026-10-09).

### Open decisions (for the supervisor meeting)

_All six implemented on 2026-10-09; see the entry above._

- **D1 - Reference set size.** Currently 30 samples on one task (proposal: 100 per task, with 50/200
  sensitivity). With 2 distinct shapes in 30, novelty is saturated and tau is degenerate.
- **D2 - Revision rounds.** Proposal says up to 3 critique-revision rounds; code runs 1.
- **D3 - Facilitator timing.** With a window of 3 and K = 5 proposals, the Facilitator can first
  fire after proposal 4, so it can change at most proposal 5. That leaves D vs D_minus little room.
  Options: smaller window, larger K, or both.
- **D4 - Token ceiling.** `TOKEN_CEILING = 40,000` is recorded in the header but not enforced.
  Either enforce it or drop it from the header.
- **D5 - Facilitator prompts.** `constraint` and `exclusion` are always filled with the generic
  "the data structure used so far" / "the current approach", because the family of earlier
  proposals is not known at run time. Works, but weaker than the proposal's examples.
- **D6 - Selection.** The Critic critiques only the last proposal; nothing chooses between the
  five. The proposal's "selection" step is logged clustering only.

### Intentional deviations from the proposal (supervisor-directed, not changed)

Recorded here so the dissertation can state them explicitly. Confirm each with the supervisor in
writing.

- Task bank: 21 tasks in 3 tiers (11 MBPP, 5 HumanEval, 5 heuristic: FunSearch/ReEvo/CO-Bench),
  instead of 24 tasks in 2 tiers (16 intro incl. ASTRA tasks, 8 APPS/heuristic).
- Diversity clustering in within-sample standardised AST space with Euclidean distance (logged in
  PREREGISTRATION.md, 2026-10-01).
- Learner conditions fixed at 1 or 2 learners (matches the proposal's 2x2 design).
