# Task Bank — Sources and Provenance

Every problem in the task bank is drawn from a published benchmark or research paper, as
required for the study. Introductory problems come from **MBPP**; intermediate algorithmic
problems from **HumanEval**; and the hard / open problems from the **FunSearch**, **ReEvo** and
**CO-Bench** line of work on LLM-discovered heuristics for NP-hard and open problems.

## References

- **MBPP** — Austin, J. et al. (2021). *Program Synthesis with Large Language Models.* arXiv:2108.07732.
- **HumanEval** — Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374.
- **FunSearch** — Romera-Paredes, B. et al. (2024). *Mathematical discoveries from program search with large language models.* Nature 625, 468–475. doi:10.1038/s41586-023-06924-6.
- **ReEvo** — Ye, H. et al. (2024). *ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution.* NeurIPS 2024. arXiv:2402.01145.
- **CO-Bench** — Sun, W. et al. (2025). *CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization.* arXiv:2504.04310.
- **APPS** (documented alternative for competition-level problems, not used in the current bank due to access limits) — Hendrycks, D. et al. (2021). *Measuring Coding Challenge Competence With APPS.* NeurIPS 2021 D&B. arXiv:2105.09938.

## Tier 1 — introductory (MBPP)

| Task id | Benchmark item | Function |
|---|---|---|
| `t1_mbpp119_search` | MBPP task 119 | `search` |
| `t1_mbpp130_max_occurrences` | MBPP task 130 | `max_occurrences` |
| `t1_mbpp140_extract_singly` | MBPP task 140 | `extract_singly` |
| `t1_mbpp167_next_power_of_2` | MBPP task 167 | `next_power_of_2` |
| `t1_mbpp19_test_duplicate` | MBPP task 19 | `test_duplicate` |
| `t1_mbpp245_max_sum` | MBPP task 245 | `max_sum` |
| `t1_mbpp256_count_primes_nums` | MBPP task 256 | `count_Primes_nums` |
| `t1_mbpp3_is_not_prime` | MBPP task 3 | `is_not_prime` |
| `t1_mbpp583_catalan_number` | MBPP task 583 | `catalan_number` |
| `t1_mbpp58_opposite_signs` | MBPP task 58 | `opposite_Signs` |
| `t1_mbpp88_freq_count` | MBPP task 88 | `freq_count` |

## Tier 2 — intermediate algorithmic (HumanEval)

| Task id | Benchmark item | Function |
|---|---|---|
| `t2_humaneval0_has_close_elements` | HumanEval/0 | `has_close_elements` |
| `t2_humaneval40_triples_sum_to_zero` | HumanEval/40 | `triples_sum_to_zero` |
| `t2_humaneval48_is_palindrome` | HumanEval/48 | `is_palindrome` |
| `t2_humaneval55_fib` | HumanEval/55 | `fib` |
| `t2_humaneval9_rolling_max` | HumanEval/9 | `rolling_max` |

## Tier 3 — hard / open (NP-hard and open problems)

These are heuristic-scored: there is no known optimal polynomial algorithm, so solutions are
judged by a metric against a baseline. This is where creative, non-obvious approaches matter most.

| Task id | Problem | Source | Function |
|---|---|---|---|
| `t3_bin_packing` | Online bin packing | FunSearch | `priority` |
| `t3_cap_set` | Cap set (open problem) | FunSearch | `build_cap_set` |
| `t3_graph_coloring` | Graph colouring order | ReEvo | `vertex_order` |
| `t3_scheduling` | Makespan scheduling | ReEvo | `priority` |
| `t3_tsp_construction` | Travelling salesman heuristic | CO-Bench / ReEvo | `choose_next` |

---

*Held-out marking tests for Tier 1/2 are the benchmark's own tests. Tier 3 has no held-out tests;
correctness is a metric computed on a frozen instance set (see each task's `heuristic:` block).*