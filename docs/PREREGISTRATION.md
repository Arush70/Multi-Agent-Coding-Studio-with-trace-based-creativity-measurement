# Pre-registration

This file records the decisions that were fixed **before** the main experiment was run and its
results were seen. The point is to stop analysis choices from being tuned to flatter the outcome.
Anything marked *provisional* is fixed on pilot data and frozen (moved to *frozen*) before the
main runs; once frozen, it is never re-tuned against results. Every change to this file is a dated
line in the log at the bottom.

_Author: Arush Kumar Vishwakarma (750096984). Supervisor: Prof. Solomon S. Oyelere._

---

## 1. Question and conditions

Does a multi-agent "studio" (Innovator / Engineer / Critic / Verifier / Facilitator over one shared
open-weight model) produce **more creative** solutions than a single-agent tutor, and does the
Facilitator's convergence-breaking add anything on top?

Five conditions, one shared model, identical task bank and budgets:

| Condition | Learners | Solver     | Facilitator |
|-----------|----------|------------|-------------|
| A         | 1        | Tutor      | –           |
| B         | 1        | Studio     | on          |
| C         | 2        | Tutor      | –           |
| D         | 2        | Studio     | on          |
| D_minus   | 2        | Studio     | **off**     |

The B–A and D–C contrasts isolate the studio; the D–D_minus contrast isolates the Facilitator.

## 2. Hypotheses / success criteria

- **SC1 (diversity):** studio conditions (B, D) show more solution families among their proposals
  than the matched tutor conditions (A, C).
- **SC2 (creativity, not just spread):** studio conditions have a higher **correct-and-novel** rate
  (passes held-out tests *and* sits in the top novelty decile vs the frozen reference).
- **SC3 (measurement validity):** automatic solution families agree with blind hand labels at
  **ARI ≥ 0.60**. This is a gate on the *instrument*, checked before any condition is compared.
- **SC4 (Facilitator):** D shows higher diversity than D_minus without a correctness collapse.

## 3. Frozen infrastructure

- **Trace schema:** JSON Schema draft 2020-12, git tag `schema-v0.1.0`. Every event validated
  before it is written; append-only; monotonic `seq`.
- **Model:** open-weight, pinned by revision (recorded in each trace header, not in this repo).
  `config/models.yaml` is never committed (it may carry an API key); keys come from the environment.
- **Reproducibility:** fixed seeds, content-addressed response cache, prompt hashes in the header.
- **Reference sets:** sampled once per task with a neutral prompt (`scripts/build_reference.py`),
  frozen, and **never** containing the studio's own outputs. Novelty and diversity are both measured
  in the space these define, so the studio is never scored against itself.

## 4. Metrics — pre-registered parameters

Two representations are analysed **separately**, never combined: interpretable **AST features**
(26-dim) and optional **UniXcoder** embeddings.

- **Novelty** = mean cosine distance to the *k* nearest neighbours in the task's frozen reference
  set, z-scored against the reference's own leave-one-out kNN distances. `k = 5`. AST features are
  standardised (per-feature) against the reference before distances are taken; embeddings are not
  (already unit-norm). "Novel" = reference percentile ≥ **0.90**.
- **Diversity** = agglomerative clustering of proposals into *families*, reported four ways
  (families in first 10, rarefaction curve, Shannon entropy, mean pairwise distance).
  **Clustering runs in the WITHIN-SAMPLE standardised AST space** (each feature z-scored across the
  session's own proposals), with **Euclidean** distance and average linkage. Diversity is measured
  *among* the proposals, so it is standardised against the proposals' own variation — deliberately a
  *different* space from novelty, which is measured *against the frozen reference*. (An earlier design
  shared the reference scaler; it was dropped because a homogeneous reference set collapsed distinct
  strategies into one family.)
- **Correctness** = pass@k on **held-out** tests (unbiased estimator, Chen et al. 2021). Held-out
  tests never enter any prompt; only the scorer reads them.
- **Correct-and-novel rate** = fraction of proposals that are both correct and above the novelty
  percentile (this is the SC2 quantity).

### Convergence / Facilitator (frozen rule)

- Convergence = novelty decay: the Facilitator fires when the window-mean nearest-neighbour distance
  falls below τ. Window = 3 proposals.
- **τ rule (amended 2026-10-15):** the 25th percentile of the Facilitator's own window statistic
  computed on 2,000 random sequences of K = 5 solutions drawn from the task's frozen reference set
  (fixed seed; `tau_from_reference_sequences`), in the space the Facilitator measures in, which is
  UniXcoder embedding space (cosine distance), chosen on the pilot (log, 2026-10-15). This is a
  *rule*, fixed in advance; the resulting number is task-specific and read off the frozen reference,
  not hand-picked. When no reference exists, or the rule still gives 0, a documented default is used
  and the session is marked as such. The original rule (leave-one-out nearest-neighbour distances
  among all reference solutions) is kept as `tau_from_reference` for comparison.
- No convergence check after the final proposal: an instruction then could not change any proposal.
- Interventions rotate deterministically: constraint → dissent → exclusion → hybridisation.

## 5. Provisional values (to be frozen by calibration before the main runs)

| Quantity                     | Provisional | How it gets frozen |
|------------------------------|-------------|--------------------|
| `CLUSTER_THRESHOLD`          | **3.0** (Euclidean, within-sample standardised AST space) | `scripts/calibrate_clustering.py`: sweep the threshold, pick the one maximising mean **ARI** vs blind hand labels (SC3). Freeze the value that meets ARI ≥ 0.60. |

**SC3 calibration protocol (run once, blind):**
1. `python scripts/calibrate_clustering.py --make-stub` → `data/calibration/labels.jsonl`
   (one row per proposal, empty `family`).
2. Fill in each `family` **without looking at the automatic clustering**, using each task's
   `known_families` vocabulary (+ `other` for genuinely new approaches). A second rater is ideal.
3. `python scripts/calibrate_clustering.py` → ARI-vs-threshold table + recommended value.
4. Copy the recommended value into `macs/analysis/aggregate.py` (`CLUSTER_THRESHOLD`), paste the
   table and date into the log below, and stop tuning it.

`known_families` in each `task.yaml` are **blind labels**: written before any model output was seen,
and never edited to match results.

## 6. Log

- _2025-…_ — schema frozen (`schema-v0.1.0`); task bank re-sourced to benchmarks (MBPP, HumanEval,
  FunSearch/ReEvo/CO-Bench), every task carries a citation and ≥3 blind `known_families`
  (see `docs/TASK_SOURCES.md`).
- _2026-09-30_ — clustering moved into the standardised AST space (was collapsing distinct strategies
  into one family on raw counts).
- _2026-10-01_ — on the first real pilot the reference-scaled cosine clustering still collapsed to one
  family (the local model's reference set has only 2 unique shapes in 30 samples). Switched diversity
  clustering to **within-sample standardisation + Euclidean distance**, decoupled from the reference;
  this gives the five pilot strategies → 6 families for both B and D_minus. `CLUSTER_THRESHOLD`
  provisional at 3.0 pending SC3.
- _2026-10-07_ — **τ rule clarified (amendment).** τ is computed in the same space the Facilitator
  measures in (cosine distance on raw AST features); it was previously read off the reference's
  standardised features, a different space. If the reference is degenerate (τ = 0, e.g. mostly
  duplicate solutions), the documented default τ = 0.05 is used. This applies the "non-degenerate
  firing rate" check already required by the proposal. Each studio trace header records `tau` and
  `tau_source`. No results had been analysed under the main design when this was changed. Also
  added: diversity compared at a matched number of solutions per task (all conditions clustered in
  one common space), reported alongside the per-session family count. Details: `docs/DEV_LOG.md`.
- _2026-10-09_ — **Session procedure brought in line with the proposal (amendment, before any
  main-run results).** (i) Facilitator window = the last three *proposals* (was: three distances,
  i.e. four proposals); τ rule unchanged. (ii) Selection: the Critic chooses one of the K = 5
  proposals (fallback: most public tests passed); critique-revision runs 1–3 rounds on it, stopping
  when a revision passes all public tests. (iii) Token ceiling of 40,000 per session enforced for
  every condition; tokens per session recorded. (iv) Reference sets: 100 per task, analysis uses
  the first 100 by seed; sensitivity at 50 and 200 via `scripts/reference_sensitivity.py`.
  (v) Facilitator instructions name the shared construct / approach. Orchestrator version 0.2.0,
  role prompts 0.2.0. Details: `docs/DEV_LOG.md`.
- _2026-10-15_ — **Model (amendment, before any main-run results).** Development and the pilot
  use `qwen3.5:9b` (Ollama id `56671c2ab938`, Apache 2.0) with thinking off
  (`reasoning_effort: none`), replacing `qwen2.5-coder:7b`. **Decision rule for the main run,
  fixed now:** if a university GPU is available by 26 October 2026, the main run uses
  Qwen3.5-27B (Apache 2.0), pinned by revision, with reference sets rebuilt and tau recomputed on
  that model and `CLUSTER_THRESHOLD` re-checked on a subset; otherwise the main run uses
  `qwen3.5:9b`. Results from different models are never pooled; the analysis enforces this.
  Details: `docs/DEV_LOG.md`.
- _2026-10-15_ — **τ rule matched to the Facilitator's statistic (amendment, before any main-run
  results).** With 100 Qwen3.5-9B reference solutions per task, 90–95% of solutions have an
  identical twin in AST-feature space, so the 25th percentile of leave-one-out nearest-neighbour
  distances was 0 on all three pilot tasks (t1_mbpp19, t1_mbpp140, t2_humaneval0) and τ fell back
  to the default. The cause is a mismatch: that rule finds the nearest of 99 other solutions, while
  the Facilitator compares a proposal with only 2–4 earlier ones. τ is now the 25th percentile of
  the Facilitator's own window statistic on random 5-solution sequences from the reference (0.050,
  0.022 and 0.003 on the three tasks). Same percentile, same frozen reference, same space; the
  statistic now matches. On the one 9B studio test session it fired at 1 of 2 checks (the
  non-degenerate firing rate the proposal requires). Also: no check after the final proposal.
  Orchestrator 0.2.1. `scripts/tau_diagnostics.py` reports both rules and firing rates; a UniXcoder
  run of it is planned before the main-run freeze. Details: `docs/DEV_LOG.md`.
- _2026-10-15_ — **Facilitator firing check on the pilot; decision rule fixed before it is
  applied.** On the first 9B pilot the matched τ rule fired on one of three tasks (AST features).
  Firing counts as non-degenerate when the would-fire rate on D_minus traces (Facilitator off) is
  between 10% and 90% of checks on every pilot task, measured by `scripts/tau_diagnostics.py` on the
  laptop. Rule, applied in order, using firing rates only (never creativity outcomes):
  (1) compute the matched τ at the 25th percentile in UniXcoder space and in AST space;
  (2) if UniXcoder meets the criterion on every pilot task, the Facilitator measures in UniXcoder
  space (the proposal's stated representation for convergence, 3.3/3.6); else if AST does, AST;
  (3) otherwise use the smallest percentile in {25, 50, 75} that meets it on every task, UniXcoder
  space first; (4) if none does, the representation and percentile with the most tasks inside the
  band are used (ties: UniXcoder before AST, then the lower percentile) and the shortfall is
  reported as a limitation of RQ2. The outcome is recorded here
  with the numbers, then frozen.
- _2026-10-15_ — **Selection and extraction robustness (amendment, pilot only).** If the Critic's
  selection reply has no valid `SELECTED: <id>`, one fixed reminder is sent before the
  most-public-tests fallback; the selection event records which path was taken. Code extraction
  accepts an unclosed code block. Orchestrator 0.2.2. Details: `docs/DEV_LOG.md`.
- _2026-10-15_ — **Facilitator check, first results (before p50/p75 are seen on the laptop).**
  At the 25th percentile neither space meets the criterion: AST fires on t1_mbpp19 only (83% of
  checks), UniXcoder on t1_mbpp19 only (17%). In UniXcoder space the studio's window values sit at
  the 96th-100th percentile of the reference's, i.e. the studio's proposals are far more spread out
  than ordinary output. Tie-break added to step (4) above, before the 50th/75th-percentile numbers
  are seen. `tau_diagnostics.py` now reports all three percentiles.
- _2026-10-15_ — **Facilitator decision (rule applied; frozen).** Laptop numbers, D_minus traces
  of the pilot, would-fire rate at the 25th / 50th / 75th percentile:
  AST — t1_mbpp140 0.00 / 0.17 / 0.83, t1_mbpp19 0.83 / 1.00 / 1.00, t2_humaneval0 0 / 0 / 0;
  UniXcoder — t1_mbpp140 0 / 0 / 0, t1_mbpp19 0.17 / 0.33 / 0.33, t2_humaneval0 0 / 0 / 0.
  No option meets the criterion on every task (steps 2 and 3); every option has exactly one task
  inside the band, so step 4 with its tie-break gives **UniXcoder space at the 25th percentile**
  (τ = 0.069 / 0.035 / 0.018 on t1_mbpp19 / t1_mbpp140 / t2_humaneval0). Frozen for the main run.
  **Limitation for RQ2, stated now:** in UniXcoder space the studio's proposals are more spread
  out than 96-100% of random sets of reference solutions, so convergence as defined (as close as
  ordinary output) is rare and the Facilitator will fire in few sessions; the D vs D_minus contrast
  is informative only where it fires, and the firing rate per task is reported with every RQ2
  result. Orchestrator 0.2.3.
- _…_ — **[fill after SC3]** `CLUSTER_THRESHOLD` frozen at __ (mean ARI __; N labelled proposals
  over __ tasks). Table:
