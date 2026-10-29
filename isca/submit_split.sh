#!/bin/bash
# Submit one model as four jobs, each on its own group of tasks (no two jobs ever share a task,
# so they never run the same session). Used for the small models on A100 MIG slices, where one
# slice is slow but many slices are free.
#
#   bash isca/submit_split.sh llama31_8b mig        # 4 jobs on 10 GB MIG slices
#   bash isca/submit_split.sh gemma3_12b gpu        # 4 jobs on full A100s
set -u
PROFILE=$1
WHERE=${2:-mig}
cd "$(dirname "$0")/.."
GROUPS=(
  "t1_mbpp119_search,t1_mbpp130_max_occurrences,t1_mbpp140_extract_singly,t1_mbpp167_next_power_of_2,t1_mbpp19_test_duplicate,t1_mbpp245_max_sum"
  "t1_mbpp256_count_primes_nums,t1_mbpp3_is_not_prime,t1_mbpp583_catalan_number,t1_mbpp58_opposite_signs,t1_mbpp88_freq_count,t2_evoeval13_find_rarest_element,t2_evoeval16_fibonacci_sum,t2_evoeval31_unique_paths"
  "t2_humaneval0_has_close_elements,t2_humaneval20_find_closest_elements,t2_humaneval26_remove_duplicates,t2_humaneval40_triples_sum_to_zero,t2_humaneval48_is_palindrome,t2_humaneval55_fib,t2_humaneval57_monotonic,t2_humaneval9_rolling_max"
  "t3_bin_packing,t3_cap_set,t3_graph_coloring,t3_scheduling,t3_tsp_construction"
)
for i in "${!GROUPS[@]}"; do
  if [ "$WHERE" = mig ]; then
    env TASKS="${GROUPS[$i]}" CONDITIONS=all SEEDS=1-6 PROFILE="$PROFILE" CONTEXT=8192 WORKERS=3 \
      GRES=gpu:1g.10gb:1 SEARCH=0 \
      sbatch -J "m-$PROFILE-g$((i + 1))" -p gpu-mig --gres=gpu:1g.10gb:1 -c 4 --mem=24G isca/run_model.sbatch
  else
    env TASKS="${GROUPS[$i]}" CONDITIONS=all SEEDS=1-6 PROFILE="$PROFILE" CONTEXT=8192 WORKERS=4 SEARCH=0 \
      sbatch -J "m-$PROFILE-g$((i + 1))" --exclude=gpu11 isca/run_model.sbatch
  fi
  sleep 20
done
