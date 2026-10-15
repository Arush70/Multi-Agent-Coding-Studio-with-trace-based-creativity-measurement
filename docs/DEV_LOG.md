# Development log

Running record of changes, fixes and open decisions, kept so the weekly logs and the
dissertation's methods chapter can be written from it. Newest entries first. Each entry says
what changed, why, and which files. Pre-registration changes are also logged, dated, in
`docs/PREREGISTRATION.md` section 6.

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
