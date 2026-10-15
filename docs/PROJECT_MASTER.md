# A Multi-Agent Coding Studio with Trace-Based Creativity Measurement

**Master project document: what we are doing, what we have done, and what we are going to do.**

- **Student:** Arush Kumar Vishwakarma (750096984), MSc Computer Science
- **Supervisor:** Prof. Solomon S. Oyelere, University of Exeter
- **Builds on:** ASTRA (a social multi-agent tutor)
- **Repository:** github.com/Arush70/Multi-Agent-Coding-Studio-with-trace-based-creativity-measurement
- **Status (build):** implementation complete for the pilot; pilot re-run, calibration and main experiment remain
- **Code version:** orchestrator 0.2.0, role prompts 0.2.0, trace schema 0.1.0
- **Last updated:** 10 October 2026

> This document is the single source of truth for the project. Aims, research questions and success
> criteria are copied from the approved proposal, so the dissertation is marked against the same
> wording. Where the build differs from the proposal, section 3 says so and why. The day-by-day
> record of changes is `docs/DEV_LOG.md`. No em dashes are used.

---

## Table of contents

1. [One-paragraph summary](#1-one-paragraph-summary)
2. [Aim, research questions and success criteria (from the proposal)](#2-aim-research-questions-and-success-criteria-from-the-proposal)
3. [Deviations from the proposal](#3-deviations-from-the-proposal)
4. [Background and motivation](#4-background-and-motivation)
5. [System architecture](#5-system-architecture)
6. [The five experimental conditions](#6-the-five-experimental-conditions)
7. [How one session runs](#7-how-one-session-runs)
8. [The trace](#8-the-trace)
9. [The sandbox](#9-the-sandbox)
10. [The task bank](#10-the-task-bank)
11. [The measurement pipeline](#11-the-measurement-pipeline)
12. [Validity and honesty safeguards](#12-validity-and-honesty-safeguards)
13. [Calibration (SC3) and human validation (SC4)](#13-calibration-sc3-and-human-validation-sc4)
14. [The dashboard](#14-the-dashboard)
15. [What we have done (status)](#15-what-we-have-done-status)
16. [Results so far](#16-results-so-far)
17. [What we are going to do (plan)](#17-what-we-are-going-to-do-plan)
18. [Timeline](#18-timeline)
19. [Repository layout](#19-repository-layout)
20. [How to run it](#20-how-to-run-it)
21. [Risks and open questions](#21-risks-and-open-questions)
22. [References](#22-references)

---

## 1. One-paragraph summary

Large language models write correct code, but they tend to give the same kind of answer again and
again, and there is no agreed way to measure how much of the space of possible solutions a coding
system explores. This project builds a Multi-Agent Coding Studio on top of ASTRA, in which an
Innovator, Engineer, Critic and Verifier are coordinated by a Facilitator that detects when ideas
start to converge and pushes for alternatives. The main contribution is a measurement pipeline that
computes novelty (distance from a reference set of typical solutions), diversity (how many solution
families were explored, and how evenly) and usefulness (correctness on held-out tests) from saved
session traces. It is tested in a 2 x 2 design (one or two simulated learners; a single Tutor or the
studio) plus a Facilitator ablation, and the measures are validated against blind human judgement.

---

## 2. Aim, research questions and success criteria (from the proposal)

**Aim.** To develop and evaluate a trace-based measurement pipeline that quantifies the creativity of
programming solutions produced in human-AI and multi-agent collaboration, tested on a Multi-Agent
Coding Studio whose Facilitator sustains rather than resolves divergence.

**Research problems.**
- **RP1 Homogenisation without an instrument:** no accepted measure of how much of the solution space a coding system explored.
- **RP2 Convergence by design:** existing multi-agent coordinators drive toward consensus; none detects and disrupts premature convergence.
- **RP3 Product-only evaluation:** benchmarks score final code by pass rate; exploration is neither logged nor measured.
- **RP4 Unknown effect of collaboration configuration:** no study crosses number of learners with number of agents and measures creativity.

**Objectives.**
- **O1** Extend ASTRA with Innovator, Engineer, Critic and Verifier agents and a Facilitator with an operational convergence rule and four intervention types.
- **O2** Extend ASTRA's JSONL trace schema so every proposal, critique, intervention and verification event is recorded with token counts and provenance.
- **O3** Implement a measurement pipeline computing novelty, diversity and usefulness from traces and code, runnable offline on saved logs.
- **O4** Run a 2 x 2 experiment (one vs two simulated learners; single Tutor vs studio with Facilitator) plus a Facilitator ablation, on introductory and harder problems.
- **O5** Validate the automated measures against blind human judgement (consensual assessment technique), supplemented by an independent LLM judge.
- **O6** Release code, prompts, traces and a reproduction notebook that needs no paid model or API access.

**Research questions.**
- **RQ1** How can the creativity of code produced in a collaborative session be operationalised from traces and artefacts (novelty, diversity, usefulness), and do the measures agree with human judgement?
- **RQ2** Does a Facilitator that detects and disrupts convergence increase the number and balance of solution families explored, relative to the same agents without it, and at what cost to correctness?
- **RQ3** How do simulated-learner configuration (one vs two) and tutor-versus-studio configuration interact in their effect on solution creativity and on the simulated learner's own contribution? (A system-level simulation, not a test of human learning.)
- **RQ4** Do the effects observed on introductory tasks hold on harder (competition-level and heuristic-design) problems where typical solutions are known to be insufficient?

**Success criteria.**

| ID | Criterion | Status |
|----|-----------|--------|
| SC1 | All configurations run end to end on at least 24 tasks, with complete schema-valid traces and no manual intervention. | Bank has 21 tasks (see section 3). Runner and traces ready. |
| SC2 | All three measures computed for every session from saved traces alone. | Met by design (`scripts/analyze.py`). |
| SC3 | Automatic family detection agrees with independent hand labels at ARI >= 0.60 on at least four tasks. | Tooling ready; labelling not started. |
| SC4 | (a) automated novelty vs blind human originality ranking (Bradley-Terry) Spearman rho >= 0.50; (b) mean pairwise Cohen's kappa across three external raters >= 0.60; (c) human vs LLM-judge agreement reported. | Ethics application submitted 10 Oct 2026; rating study after approval. |
| SC5 | For each primary outcome: effect, 95% CI, model specification and minimum detectable effect, with adjusted p-values for planned contrasts. | Not started (needs main run). |
| SC6 | All statistics recomputable from released traces on a CPU; full experiment rerunnable without paid model or API access. | On track (open models, cache, CPU analysis). |

---

## 3. Deviations from the proposal

Stated openly so examiners see considered decisions, not drift. Each is also in `docs/DEV_LOG.md`.

| Proposal | Build | Reason / decision |
|---|---|---|
| 24 tasks in 2 tiers (16 introductory incl. ASTRA tasks; 8 APPS competition + heuristic) | 21 tasks in 3 tiers: 11 MBPP, 5 HumanEval, 5 heuristic-design (FunSearch, ReEvo, CO-Bench) | Supervisor asked for every task to come from a published, citable benchmark. APPS host was unreachable from the build environment. **To confirm with supervisor in writing; SC1 says 24, so either add 3 tasks or record the amendment.** |
| Tier 2 = competition-level | HumanEval (standard) as Tier 2; heuristic problems as Tier 3 | RQ4 wording ("competition-level") to be reconciled with the bank. |
| Clustering with a calibrated threshold (distance unspecified) | Within-session standardised AST features, Euclidean distance, average linkage; plus a matched-sample comparison across conditions | Earlier versions collapsed distinct strategies into one family. Logged in PREREGISTRATION.md (1 Oct and 7 Oct 2026). |
| UniXcoder as a parallel primary analysis | Implemented (`represent.embed`), not yet run | Planned for the main run on GPU. |
| Mutation testing on five tasks; learner-contribution metric | Not built yet | In the plan (section 17). |
| 6 repetitions x 5 cells x 24 tasks = 720 sessions | Batch runner ready; pilot first | Main run after pilot and SC3. |

---

## 4. Background and motivation

The project builds on **ASTRA**, a social multi-agent tutor that compares learner configurations
(alone with a tutor, pair with a tutor, pair with a tutor and facilitator), logs every turn to a
JSONL trace and derives participation measures. This project keeps the social-learning idea and the
trace-based approach, but changes the outcome being measured: instead of participation, it measures
the **creativity of the code produced**, and adds the instrument ASTRA does not have.

Why it matters: LLMs and preference tuning narrow outputs toward typical answers (Kirk et al., 2024);
writers helped by AI produced better individual stories but a less diverse set (Doshi and Hauser,
2024); multi-agent coordinators are built to reach consensus (Du et al., 2024; Hong et al., 2024);
and evaluation scores only final pass rate (Chen et al., 2021). Whether a team of agents explores
more of the solution space, and whether a Facilitator helps, has not been measured.

---

## 5. System architecture

Everything runs on a **single open-weight model**; agents are the same model with different role
prompts (`macs/prompts/roles.yaml`, version 0.2.0, hashes written into every trace). Development
uses `qwen2.5-coder:7b-instruct-q4_K_M` through Ollama; the main run will use a larger open model
on GPU. **No humans act during a session.**

### 5.1 The agents

- **Learners:** simulated personas that take the first turn. Their turns are **passed to the
  solver** (the Tutor or the Innovator); in two-learner conditions, learner 2 sees learner 1's turn.
- **Tutor (baseline):** writes one solution.
- **Studio:**
  - **Innovator:** proposes a strategy that differs from those already proposed (K = 5 proposals).
  - **Engineer:** implements each strategy, and later revises the selected one.
  - **Verifier:** runs the public tests in the sandbox and reports the result.
  - **Critic:** first **selects** which of the five proposals to develop (seeing strategy, code and
    test result for each), then critiques the code in each revision round.
  - **Facilitator:** detects convergence and injects one of four interventions.

### 5.2 Learner personas

Six fixed personas (`macs/prompts/personas.yaml`) from ASTRA's synthetic-learner dimensions: prior
knowledge, confidence and how often they challenge the agent. Selected deterministically from the
seed (`start = seed % 6`, learner j gets `(start + j) % 6`), so runs are reproducible.

| ID | Prior knowledge | Confidence | Challenges |
|----|-----------------|------------|------------|
| L01 | low | low | rarely |
| L02 | low | high | often |
| L03 | medium | medium | sometimes |
| L04 | high | medium | often |
| L05 | medium | low | rarely |
| L06 | high | high | often |

---

## 6. The five experimental conditions

| Condition | Learners | Solver | Facilitator |
|-----------|----------|--------|-------------|
| **A** | 1 | Tutor | - |
| **B** | 1 | Studio | on |
| **C** | 2 | Tutor | - |
| **D** | 2 | Studio | on |
| **D_minus** | 2 | Studio | **off** |

- **B vs A, D vs C:** the tutor-versus-studio effect (a change of roles, orchestration and Facilitator together).
- **A vs C, B vs D:** the simulated-learner effect (RQ3).
- **D vs D_minus:** the Facilitator effect (RQ2).

The learner count is 1 or 2, matching the proposal's 2 x 2 design. The engine loops over
`cfg.n_learners`, so N learners is possible as an optional extension.

---

## 7. How one session runs

1. **Learners** take their turns (learner 2 sees learner 1).
2. **Tutor path (A, C):** the Tutor sees the task plus the learners' turns and writes one solution;
   the Verifier runs the public tests.
3. **Studio path (B, D, D_minus):**
   1. **Divergence:** five rounds of Innovator -> Engineer -> Verifier. The Innovator sees the task,
      the learners' turns and all strategies so far. After each proposal the Facilitator checks for
      convergence (7.1); if it fires, its instruction is added to the next Innovator turn.
   2. **Selection:** the Critic sees all five proposals (strategy, code, public-test result) and
      replies `SELECTED: pN`. If the reply is unusable, the proposal passing the most public tests is
      taken (earliest on a tie) and the trace says `fallback`.
   3. **Critique-revision (1 to 3 rounds):** the Critic critiques the selected code; the Engineer
      revises it (seeing the code and the critique); the Verifier tests it. The loop stops once a
      revision passes every public test, otherwise after three rounds. Each revision builds on the
      previous one (`p_rev1 -> p_rev2 -> p_rev3`).
4. **Token ceiling:** every model call is counted. When the next call would start at or above
   40,000 prompt + completion tokens, the session stops cleanly. This applies to every condition.
5. Everything is written to the trace and validated.

A studio session yields 5 proposals plus 1 to 3 revisions; a tutor session yields 1 solution.
Because of this, diversity is compared at a **matched number of solutions** (section 11.2).

### 7.1 The Facilitator (frozen rule)

- **Convergence = novelty decay.** Each new proposal's distance to its nearest earlier proposal is
  computed (cosine distance in UniXcoder embedding space, chosen on the 15 Oct pilot; needs
  `pip install torch transformers`). The Facilitator fires when the mean of these
  distances over the last **three proposals** falls below tau. It can first fire after proposal 3,
  so it can shape proposals 4 and 5. There is no check after proposal 5, since an instruction then
  could not change anything.
- **tau rule (amended 15 Oct):** the 25th percentile of the Facilitator's own window statistic on
  2,000 random sequences of 5 solutions drawn from the frozen reference set (same space, fixed
  seed). The earlier rule (leave-one-out nearest-neighbour distances among all 100 reference
  solutions) gave 0 on every pilot task because most solutions have an identical twin among 99
  others. If the rule still gives 0, or there is no reference, the documented default 0.05 is used.
  Each trace header records `tau`, `tau_source`, `tau_rule` and `facilitator_representation`.
  Check with `python scripts/tau_diagnostics.py --representation unixcoder`. On the pilot it fired
  on one of three tasks in every space and percentile tried; this is a stated limitation of RQ2.
- **Interventions rotate:** constraint -> dissent -> exclusion -> hybridisation. They are specific:
  the Facilitator names the construct most of the recent implementations share (a set, a dictionary,
  sorting, recursion, a heap, bit manipulation, nested loops) and the latest approach name, e.g.
  "Propose a strategy that does NOT use a set."

---

## 8. The trace

- One **append-only JSONL file** per session in `data/traces/`, every line validated against a JSON
  Schema (draft 2020-12, frozen as git tag `schema-v0.1.0`) before it is written.
- Event fields: `session_id`, `task_id`, `condition`, `seq`, `ts`, `role`, `event`, `content`,
  token counts, request hash, proposal and parent ids, intervention type, trigger metric, threshold,
  test results.
- **Header** records model, pinned revision, quantisation, temperature, seed, prompt hashes,
  `orchestrator_version`, `n_learners`, token ceiling, max revision rounds and, for studio sessions,
  `tau`, `tau_source`, `k_proposals`, `convergence_window` and `reference_n`.
- **session_end** records `completed; tokens_used=...` or `stopped: token ceiling ...`.
- All measurement runs afterwards on saved traces, never during generation.

---

## 9. The sandbox

- **subprocess** (development): separate process with a timeout; CPU and memory limits on POSIX only.
  Not a security boundary.
- **docker** (reported results): `--network none`, 512 MB memory, 1 CPU, read-only filesystem,
  non-root user, PID limit.
- A wrong answer is data; a crash or timeout is flagged separately.

---

## 10. The task bank

21 problems in three tiers, each traceable to a published source, with public tests (shown to
agents), held-out tests (scorer only) and blind `known_families` written before any model output.

| Tier | What | Count | Source |
|------|------|-------|--------|
| 1 | MBPP coding tasks | 11 | Austin et al. 2021 |
| 2 | HumanEval coding tasks | 5 | Chen et al. 2021 |
| 3 | Bin packing, cap set, TSP, graph colouring, scheduling (heuristic design) | 5 | FunSearch (Nature 2024), ReEvo (NeurIPS 2024), CO-Bench (2025) |

- **Tier 1 / 2:** the model writes the solution; graded pass/fail on held-out tests.
- **Tier 3:** the model writes a heuristic rule; the harness applies it to frozen instances and
  scores it against a standard baseline (score > 1 beats it; infeasible = 0).

See section 3 for how this differs from the proposal.

---

## 11. The measurement pipeline

```
Solutions (code)
  -> normalise + represent (26 AST features; UniXcoder as a parallel analysis)
      -> diversity   (families: per session, and at a matched sample size across conditions)
      -> novelty     (kNN distance from a frozen reference set of 100)
      -> correctness (held-out tests / heuristic score)
          -> correct-and-novel rate
              -> per condition: table, charts, mixed-effects models
```

### 11.1 Representation
Code is normalised (comments and docstrings stripped, local names canonicalised, black formatting),
then turned into 26 AST features. UniXcoder embeddings are analysed separately, never merged.

### 11.2 Diversity
- **Per session:** families among a session's own solutions (within-session standardised features,
  Euclidean, average linkage, threshold 3.0 provisional until SC3), entropy, rarefaction, mean
  pairwise distance.
- **Matched comparison (main tutor-vs-studio measure):** for each task, all solutions from all
  conditions and seeds are clustered together in one common standardised space, and each condition's
  family count is read off its rarefaction curve at the **same number of solutions m**. This is
  needed because a tutor session has one solution and a studio session several, so the per-session
  count is capped at 1 for tutors by construction.

### 11.3 Novelty
- Reference set: **100 solutions per task**, sampled once from the same model with a neutral prompt
  at temperature 0.7 and fixed seeds, then frozen. The analysis always uses the first 100 by seed.
- Novelty = mean cosine distance to the k = 5 nearest reference solutions, z-scored against the
  reference's own leave-one-out distances. "Novel" = reference percentile >= 0.90.
- **Sensitivity:** `scripts/reference_sensitivity.py` compares 50 / 100 / 200 (Spearman rank
  correlation and agreement of the novel flag).

### 11.4 Correctness
pass@k on held-out tests (unbiased estimator, Chen et al. 2021); Tier 3 by heuristic score.

### 11.5 Creativity outcome
The correct-and-novel rate: share of solutions that pass held-out tests **and** are in the top
novelty decile. No weighted composite. Wrong solutions are kept for diversity and count as zero here.

### 11.6 Cost
Tokens per session are reported per condition with every contrast (proposal 3.1).

---

## 12. Validity and honesty safeguards

- Held-out tests never enter a prompt; only the scorer reads them.
- Reference sets exclude the studio's own outputs and are frozen.
- `known_families` are blind labels written before any output.
- Pre-registration (`docs/PREREGISTRATION.md`) fixes thresholds before results; every change is a dated amendment.
- Two representations analysed separately.
- Pinned model revision, fixed seeds, content-addressed response cache, prompt hashes in every header.
- `orchestrator_version` in every trace; the analysis skips traces from older code and crashed
  sessions by default, so results never mix incomparable data.
- Null results and minimum detectable effects are reported.

---

## 13. Calibration (SC3) and human validation (SC4)

**SC3: family detection vs hand labels (ARI >= 0.60, at least four tasks).**
1. `python scripts/calibrate_clustering.py --make-stub` writes `data/calibration/labels.jsonl`.
2. Label each solution blind (no clustering, no condition), using the task's `known_families` plus `other`.
3. A second independent rater (supervisor or PhD student) labels the same set; report Cohen's kappa.
4. `python scripts/calibrate_clustering.py` sweeps the threshold, maximising ARI against the consensus.
5. Freeze `CLUSTER_THRESHOLD`, record table and date in PREREGISTRATION.md, stop tuning.

**SC4: external human validation (after ethics approval).**
Three volunteer postgraduate students (MSc or PhD) in Computer Science with Python experience, not
the author or supervisor and not taught or assessed by them. Each compares the same 60 pairs of
correct solutions (about one hour, online, blind to condition): "which takes the more original
approach?" and "which is the more creative solution?". Aggregated with Bradley-Terry; agreement by
mean pairwise Cohen's kappa; Spearman rho against automated novelty. A DeepSeek LLM judge answers
the same questions and is reported separately. Data are pseudonymised under participant codes and
anonymised after the withdrawal deadline. **Ethics application submitted on Worktribe, 10 October 2026.**

---

## 14. The dashboard

Streamlit app (`streamlit run macs/ui/app.py`), four tabs: Overview (condition comparison using
families at a matched sample size, novelty, correct-and-novel, tokens), How creativity is measured,
Traces (per-proposal strategy, code, correctness, novelty, creative flag, interventions, timeline),
and Run.

---

## 15. What we have done (status)

**Built and tested (62 automated tests):**
- Trace schema, validating writer, git-tagged freeze.
- Task bank of 21 sourced tasks with blind labels and held-out tests; provenance in `docs/TASK_SOURCES.md`.
- Orchestrator for all five conditions, with learners feeding the solver, Critic selection, 1 to 3
  revision rounds, enforced token ceiling, and the Facilitator (window of three proposals, tau rule
  with degenerate fallback, specific interventions).
- Sandbox (subprocess and Docker).
- Metrics: representation, novelty, diversity (per session and matched), correctness, creativity,
  Tier 3 heuristic scoring.
- Reference builder (100 per task, top-up, resumable) and sensitivity script.
- Batch runner `scripts/run_experiment.py` (tasks x conditions x seeds, resumable).
- Analysis `scripts/analyze.py` (results table with tokens and interventions; matched diversity).
- Calibration tooling, pre-registration, development log, dashboard.
- Ethics application submitted (10 October 2026).

**Week 3 code review (7 and 9 October):** found that learner turns never reached the solver, the
Facilitator's tau was 0 on the pilot task, the Critic and revision never saw the code, and the
"1 vs 6 families" comparison was true by construction. All fixed; six further changes brought the
session procedure in line with the proposal. Details in `docs/DEV_LOG.md`.

---

## 16. Results so far

**The Week 2 pilot result ("tutors 1 family, studios 6") is withdrawn.** The traces showed that A
and C produced byte-identical code, and B, D and D_minus produced byte-identical code, because the
learners' turns were never passed to the solver; the Facilitator could not fire on that task; and a
tutor session can only ever have one family. Those traces are kept but are excluded from the
analysis (older orchestrator version).

**Next result:** a pilot on three tasks (one per tier) x five conditions x three seeds with the fixed
code and 100-sample references.

---

## 17. What we are going to do (plan)

Priorities: **now** = before the next supervision, **soon** = this month, **later** = before submission.

### A. Pilot and calibration
- [ ] Build 100-sample references for the pilot tasks. *(now)*
- [ ] Re-run the pilot: 3 tasks x 5 conditions x 3 seeds. *(now)*
- [ ] Check the pilot: Facilitator firing rate, selection fallback rate, revision rounds, tokens. *(soon)*
- [ ] SC3: hand-label four tasks, second rater, freeze `CLUSTER_THRESHOLD`. *(soon)*
- [ ] Build references for all 21 tasks; sensitivity at 50 / 200 on a subset. *(soon)*

### B. Main experiment
- [ ] Decide the main-run model (stronger open model on GPU / Kaggle). *(soon)*
- [ ] Main run: all tasks x 5 conditions x 6 seeds on the Docker sandbox. *(soon)*
- [ ] Run UniXcoder as the parallel representation. *(later)*

### C. Remaining measures
- [ ] Learner-contribution metric (share of final-solution lineage from learner turns; novelty of learner ideas). *(soon)*
- [ ] Mutation testing (mutmut) on the held-out suites of five tasks. *(later)*

### D. Statistics
- [x] Mixed-effects models (learner x solver, task random effect), planned contrasts B-A, D-C, D-D_minus, Holm-adjusted, 95% CIs, minimum detectable effect from pilot variance. Done 15 Oct: `macs/analysis/stats.py`, `scripts/stats_report.py`.

### E. Human validation
- [ ] On ethics approval: recruit raters, build 60 pairs, run the rating study, LLM judge, analysis. *(later)*

### F. Dissertation
- [ ] Draft chapters; figures and tables from the analysis. *(later)*

### G. Dashboard polish (from the Week 2 review)
- [ ] Show the value each solution returns for the example inputs; plain-language captions; "Clear traces" button; auto-conclusion after re-score. *(soon)*

---

## 18. Timeline

- **Weeks 1-2 (21 Sep - 4 Oct):** foundation, engine, measurement pipeline, dashboard, first pilot.
- **Week 3 (5 - 11 Oct):** code review and fixes, proposal alignment, batch runner, ethics application submitted.
- **Weeks 4-5 (12 - 25 Oct):** pilot re-run, SC3 calibration, references for all tasks, freeze and pre-register.
- **Weeks 6-7 (26 Oct - 8 Nov):** main experiment.
- **Weeks 8-9 (9 - 22 Nov):** rating study, mixed-effects and sensitivity analysis, results chapters.
- **Week 10 (23 - 29 Nov):** revision, notebook, release.
- **Dissertation submission:** 30 November 2026.

---

## 19. Repository layout

```
macs/
  orchestrator.py         sessions: learners, tutor, studio (selection, revisions, token ceiling)
  facilitator.py          convergence rule, interventions, construct/approach naming
  llm/client.py           model client with content-addressed cache
  prompts/                roles.yaml (v0.2.0), personas.yaml
  sandbox/runner.py       subprocess + docker backends
  tasks/bank.py           task loader (held-out tests only for the scorer)
  trace/                  writer (validated, end_note) + schema.json
  metrics/                represent, novelty, diversity, usefulness, reference (REFERENCE_N = 100)
  analysis/               score, heuristic_score, aggregate (per session + pooled_diversity)
  ui/app.py               Streamlit dashboard
tasks/tier1|tier2|tier3/  21 problems
scripts/
  build_reference.py      reference sets (top-up, resumable)
  run_experiment.py       batch runner (tasks x conditions x seeds, resumable)
  run_session.py          one session
  analyze.py              results table + summary.json
  reference_sensitivity.py  50 / 100 / 200 check
  calibrate_clustering.py SC3 stub + ARI sweep
  make_instances.py       Tier 3 instances
tests/                    62 tests
docs/                     PREREGISTRATION.md, DEV_LOG.md, TASK_SOURCES.md, PROJECT_MASTER.md
data/                     reference/, traces/, results/, calibration/ (not committed)
```

---

## 20. How to run it

```powershell
# setup (once)
python -m venv .venv ; .venv\Scripts\Activate.ps1 ; pip install -e ".[dev]"
ollama pull qwen2.5-coder:7b-instruct-q4_K_M ; ollama serve
copy config\models.example.yaml config\models.yaml
python -m pytest -q

# 1. references (100 per task; tops up existing files; safe to stop and restart)
python scripts\build_reference.py --tasks t1_mbpp19_test_duplicate,t2_humaneval0_has_close_elements,t3_bin_packing

# 2. sessions (resumable; --dry-run shows the plan)
python scripts\run_experiment.py --tasks t1_mbpp19_test_duplicate,t2_humaneval0_has_close_elements,t3_bin_packing --seeds 1-3

# 3. results
python scripts\analyze.py

# dashboard
streamlit run macs\ui\app.py
```

Main run: `--tasks all --seeds 1-6 --sandbox docker`.

---

## 21. Risks and open questions

- **Weak development model.** The 7B model may give near-identical reference solutions even at 100,
  so tau falls back to the default and novelty saturates. Mitigation: stronger model for the main run;
  stated as a limitation if it persists.
- **Task bank vs SC1.** 21 tasks against "at least 24"; decide with the supervisor.
- **Compute.** About 80 GPU-hours estimated for the main run; Kaggle quota plus laptop for tutor
  cells; repetitions drop to 4 if short (proposal fallback).
- **SC3 second rater** needed for the rigorous version.
- **Ethics timing.** The rating study cannot start until approval.
- **Simulation.** The learner factor measures personas, not students; no claims about real learners.

---

## 22. References

- Austin et al. (2021), *Program Synthesis with Large Language Models*, arXiv:2108.07732.
- Chen et al. (2021), *Evaluating Large Language Models Trained on Code*, arXiv:2107.03374.
- Chakrabarty et al. (2024), *Art or Artifice? Large Language Models and the False Promise of Creativity*, CHI 2024.
- Doshi and Hauser (2024), *Generative AI enhances individual creativity but reduces the collective diversity of novel content*, Science Advances 10(28).
- Du et al. (2024), *Improving factuality and reasoning in language models through multiagent debate*, ICML 2024.
- Hong et al. (2024), *MetaGPT: Meta programming for a multi-agent collaborative framework*, ICLR 2024.
- Kirk et al. (2024), *Understanding the effects of RLHF on LLM generalisation and diversity*, ICLR 2024.
- Oyelere (2026), *ASTRA: A synthetic benchmark for trace-based evaluation of socially intelligent multi-agent tutoring and participation-balanced collaboration in introductory programming*, Computers and Education: Artificial Intelligence, 100633, doi:10.1016/j.caeai.2026.100633.
- Romera-Paredes et al. (2024), *Mathematical discoveries from program search with large language models*, Nature 625:468-475.
- Ye et al. (2024), *ReEvo*, NeurIPS 2024, arXiv:2402.01145.
- Sun et al. (2025), *CO-Bench*, arXiv:2504.04310.
- Hendrycks et al. (2021), *APPS* (considered, not used), arXiv:2105.09938.

---

*Keep this file updated. Detailed, dated changes go in `docs/DEV_LOG.md`.*
