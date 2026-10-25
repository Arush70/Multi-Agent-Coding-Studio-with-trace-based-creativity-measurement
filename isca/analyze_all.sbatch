#!/bin/bash
#SBATCH -A COMM514
#SBATCH -p sq
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -J m-analysis
#SBATCH -o data/jobs/analysis-%j.log
#SBATCH --export=ALL
#SBATCH -D .
#
# The whole analysis, on a CPU node (no GPU needed), for every model that has traces. Safe to run
# again whenever more sessions have finished; it rewrites the results from scratch.
#
#   sbatch isca/analyze_all.sbatch
#
# Main model (Qwen3.5-27B, pre-registered 15 Oct): data/results/summary.json, stats.json,
# stats_contamination.json, frontier.json/png, mutation.json, summary_unixcoder.json.
# Every model: data/results/models/<model>/summary.json and stats.json.
# Across models: data/results/model_consistency.json. Open problems: search_report.json.
# Everything is packed into data/jobs/analysis_results.tgz to download.
set -uo pipefail
MAIN=${MAIN:-qwen3.5:27b}
PREFIX=${MACS_PREFIX:-$HOME/macs-tools}
source "$PREFIX/miniforge3/etc/profile.d/conda.sh"
conda activate macs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HOME="$PREFIX/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "== analysis job $SLURM_JOB_ID on $(hostname) $(date)"
mkdir -p data/results/models
python -c "import matplotlib" 2>/dev/null || pip install -q matplotlib     # for the frontier chart

MODELS=$(python -c "from macs.analysis.aggregate import trace_models; print(' '.join(sorted({m.split('@')[0] for m in trace_models('data/traces')})))")
echo "models: $MODELS"
slug() { python -c "from macs.metrics.reference import model_slug; print(model_slug('$1'))"; }

# 1. one analysis per model, in parallel (each is single-threaded)
for m in $MODELS; do
  s=$(slug "$m"); mkdir -p "data/results/models/$s"
  python scripts/analyze.py --model "$m" --out "data/results/models/$s/summary.json" > "data/results/models/$s/analyze.log" 2>&1 &
done
# UniXcoder as a parallel representation, main model only
python scripts/analyze.py --model "$MAIN" --representation unixcoder --out data/results/summary_unixcoder.json \
  > data/results/analyze_unixcoder.log 2>&1 &
wait
echo "== per-model analysis done $(date)"

# 2. statistics per model (full design: 6 seeds; panel: 3 seeds)
for m in $MODELS; do
  s=$(slug "$m"); seeds=3
  case $m in qwen3.5:27b|qwen3.5:9b) seeds=6 ;; esac
  python scripts/stats_report.py --summary "data/results/models/$s/summary.json" --tasks 24 --seeds $seeds \
    --out "data/results/models/$s/stats.json" > "data/results/models/$s/stats.log" 2>&1
done

# 3. the main model's results in the places the dashboard reads
S=$(slug "$MAIN")
cp "data/results/models/$S/summary.json" data/results/summary.json
python scripts/stats_report.py --tasks 24 --seeds 6 | tee data/results/stats.log
python scripts/stats_report.py --task-set contamination_check --out data/results/stats_contamination.json \
  > data/results/stats_contamination.log 2>&1
python scripts/frontier.py --plot > data/results/frontier.log 2>&1 || true
python scripts/mutation_test.py --model "$MAIN" > data/results/mutation.log 2>&1 || true

# 4. across models, and the open-problem search
python scripts/model_consistency.py | tee data/results/model_consistency.log
python scripts/search_open.py report > data/results/search_report.log 2>&1 || true

tar -czf data/jobs/analysis_results.tgz data/results
echo "== analysis finished $(date); download data/jobs/analysis_results.tgz"
