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
  **Clustering runs in the same standardised AST space as novelty** — the frozen reference scaler
  when the task has a reference set, otherwise within-sample scaling (pilot only, flagged in output).
  Distance metric: cosine, linkage: average.
- **Correctness** = pass@k on **held-out** tests (unbiased estimator, Chen et al. 2021). Held-out
  tests never enter any prompt; only the scorer reads them.
- **Correct-and-novel rate** = fraction of proposals that are both correct and above the novelty
  percentile (this is the SC2 quantity).

### Convergence / Facilitator (frozen rule)

- Convergence = novelty decay: the Facilitator fires when the window-mean nearest-neighbour distance
  falls below τ. Window = 3 proposals.
- **τ rule:** the 25th percentile of the reference set's leave-one-out nearest-neighbour distances
  (`tau_from_reference`). This is a *rule*, fixed in advance; the resulting number is task-specific
  and read off the frozen reference, not hand-picked. When no reference exists, a documented default
  is used and the session is marked as such.
- Interventions rotate deterministically: constraint → dissent → exclusion → hybridisation.

## 5. Provisional values (to be frozen by calibration before the main runs)

| Quantity                     | Provisional | How it gets frozen |
|------------------------------|-------------|--------------------|
| `CLUSTER_THRESHOLD`          | **0.30** (standardised AST space, cosine) | `scripts/calibrate_clustering.py`: sweep the threshold, pick the one maximising mean **ARI** vs blind hand labels (SC3). Freeze the value that meets ARI ≥ 0.60. |

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
- _2025-…_ — clustering moved into the standardised AST space (was collapsing distinct strategies
  into one family on raw counts). `CLUSTER_THRESHOLD` remains provisional at 0.30 pending SC3.
- _…_ — **[fill after SC3]** `CLUSTER_THRESHOLD` frozen at __ (mean ARI __; N labelled proposals
  over __ tasks). Table:
