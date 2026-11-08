#!/bin/bash
# Submit one model as four jobs, each on its own group of tasks (no two jobs ever share a task,
# so they never run the same session). Used for the small models on A100 MIG slices, where one
# slice is slow but many slices are free.
#
#   bash isca/submit_split.sh llama31_8b mig        # 4 jobs on 10 GB MIG slices (3 sessions each)
#   WORKERS=1 bash isca/submit_split.sh granite42_8b mig   # 1 session each, when 3 do not fit in 10 GB
#   SEEDS=1-3 WORKERS=1 bash isca/submit_split.sh gemma3_12b mig   # with SEEDS=4-6 as well: 8 jobs that never
#                                                                  # share a session (unit = task and seed)
#   bash isca/submit_split.sh gemma3_12b gpu        # 4 jobs on full A100s (context 16384, enough for every profile)
#   EXTRA="--gres=gpu:2 -c 24 --mem=128G" bash isca/submit_split.sh qwen35_122b gpu   # two-GPU models
#   GROUPS_ONLY=2,3,4 ...  submits only those task groups
#   CONTEXT=8192 WORKERS=1 EXTRA="--gres=gpu:2 -c 24 --mem=200G" bash isca/submit_split.sh deepseek_v2_236b gpu   # 236B: one 8192-token session per job, so all 61 layers fit on the two GPUs
set -u
PROFILE=$1
WHERE=${2:-mig}
cd "$(dirname "$0")/.."
TASK_GROUPS=(
  "t1_mbpp119_search,t1_mbpp130_max_occurrences,t1_mbpp140_extract_singly,t1_mbpp167_next_power_of_2,t1_mbpp19_test_duplicate,t1_mbpp245_max_sum"
  "t1_mbpp256_count_primes_nums,t1_mbpp3_is_not_prime,t1_mbpp583_catalan_number,t1_mbpp58_opposite_signs,t1_mbpp88_freq_count,t2_evoeval13_find_rarest_element,t2_evoeval16_fibonacci_sum,t2_evoeval31_unique_paths"
  "t2_humaneval0_has_close_elements,t2_humaneval20_find_closest_elements,t2_humaneval26_remove_duplicates,t2_humaneval40_triples_sum_to_zero,t2_humaneval48_is_palindrome,t2_humaneval55_fib,t2_humaneval57_monotonic,t2_humaneval9_rolling_max"
  "t3_bin_packing,t3_cap_set,t3_graph_coloring,t3_scheduling,t3_tsp_construction"
)
for i in "${!TASK_GROUPS[@]}"; do
  # GROUPS=2,3,4 submits only those task groups (1-4), e.g. when group 1 is already running
  if [ -n "${GROUPS_ONLY:-}" ] && [[ ",$GROUPS_ONLY," != *",$((i + 1)),"* ]]; then continue; fi
  if [ "$WHERE" = mig ]; then
    env TASKS="${TASK_GROUPS[$i]}" CONDITIONS=all SEEDS="${SEEDS:-1-6}" PROFILE="$PROFILE" CONTEXT=8192 WORKERS="${WORKERS:-3}" \
      GRES=gpu:1g.10gb:1 SEARCH=0 \
      sbatch -J "m-$PROFILE-g$((i + 1))-s${SEEDS:-1-6}" -p gpu-mig --gres=gpu:1g.10gb:1 -c 4 --mem=24G isca/run_model.sbatch
  else
    env TASKS="${TASK_GROUPS[$i]}" CONDITIONS=all SEEDS=1-6 PROFILE="$PROFILE" CONTEXT="${CONTEXT:-16384}" WORKERS="${WORKERS:-4}" SEARCH=0 \
      sbatch -J "m-$PROFILE-g$((i + 1))" --exclude=gpu11 ${EXTRA:-} isca/run_model.sbatch
  fi
  sleep 20
done
