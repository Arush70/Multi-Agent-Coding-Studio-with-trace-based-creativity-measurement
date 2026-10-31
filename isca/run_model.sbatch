#!/bin/bash
#SBATCH -A COMM514
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 23:50:00
#SBATCH -J macs
#SBATCH -o data/jobs/isca-%j.log
#SBATCH --export=ALL
#SBATCH -D .
#
# Runs the experiment for one model on one full A100 (80 GB), submitted from the repository folder.
# A GPU node has 2 A100s and 24 CPUs, so a job asks for 12 CPUs and two jobs fit on one node:
#
#   sbatch isca/run_model.sbatch                                   # main run, Qwen3.5-27B
#   PROFILE=gpt_oss_20b CONDITIONS=A,B,C,D,D_minus,E,E_minus SEEDS=1-3 TASKS=main sbatch isca/run_model.sbatch
#
# For each task it builds the 100 reference solutions (tau and novelty come from them), then runs
# the sessions. WORKERS sessions run at the same time against one Ollama server, which keeps the
# GPU busy. Everything is resumable: finished sessions are skipped. The job queues its own
# follow-up (up to RESUBMIT times) for when the 24-hour limit is reached, and the follow-up exits
# at once if nothing is left. Afterwards it runs the open-problem search on the same model.
set -uo pipefail
PROFILE=${PROFILE:-qwen35_27b}
TASKS=${TASKS:-all}
CONDITIONS=${CONDITIONS:-all}
SEEDS=${SEEDS:-1-6}
WORKERS=${WORKERS:-6}
SEARCH=${SEARCH:-auto}
# the open-problem search is pre-registered on the main model; it also runs on 9B for comparison
if [ "$SEARCH" = auto ]; then case $PROFILE in qwen35_27b|local) SEARCH=1 ;; *) SEARCH=0 ;; esac; fi
RESUBMIT=${RESUBMIT:-3}
PREFIX=${MACS_PREFIX:-$HOME/macs-tools}

source "$PREFIX/miniforge3/etc/profile.d/conda.sh"
conda activate macs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_HOME="$PREFIX/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OLLAMA_MODELS="$PREFIX/models" OLLAMA_HOST="127.0.0.1:$((11500 + SLURM_JOB_ID % 400))"
export OLLAMA_NUM_PARALLEL=$WORKERS OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_CONTEXT_LENGTH=${CONTEXT:-16384}  # at least the profile's context_length (checked by run_experiment); CONTEXT=8192 saves GPU memory
OLLAMA="$PREFIX/ollama/bin/ollama"     # replaced by a node-local copy below
scontrol update JobId="$SLURM_JOB_ID" JobName="m-$PROFILE" 2>/dev/null || true
echo "== job $SLURM_JOB_ID on $(hostname) $(date) profile=$PROFILE tasks=$TASKS conditions=$CONDITIONS seeds=$SEEDS workers=$WORKERS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

remaining() {
  python scripts/run_experiment.py --tasks "$TASKS" --conditions "$CONDITIONS" --seeds "$SEEDS" \
    --profile "$PROFILE" --dry-run 2>/dev/null | grep -oE "[0-9]+ to run" | grep -oE "[0-9]+"
}
# refuse a bad task, condition or profile list before anything else (and before queuing a follow-up)
if ! python scripts/run_experiment.py --tasks "$TASKS" --conditions "$CONDITIONS" --seeds "$SEEDS" \
     --profile "$PROFILE" --dry-run > /dev/null 2> "data/jobs/check-$SLURM_JOB_ID.err"; then
  echo "invalid job settings (TASKS=$TASKS CONDITIONS=$CONDITIONS SEEDS=$SEEDS PROFILE=$PROFILE):"
  cat "data/jobs/check-$SLURM_JOB_ID.err"; exit 1
fi
LEFT=$(remaining); LEFT=${LEFT:-1}
DONE_SEARCH=data/jobs/search-done-$PROFILE
if [ "$LEFT" = 0 ] && { [ "$SEARCH" != 1 ] || [ -f "$DONE_SEARCH" ]; }; then
  echo "nothing left to run for $PROFILE"; exit 0
fi
if [ "$RESUBMIT" -gt 0 ]; then
  # the follow-up asks for the same GPUs, CPUs and memory as this job (e.g. --gres=gpu:2 for 122B)
  sbatch --dependency=afterany:"$SLURM_JOB_ID" --export=ALL,RESUBMIT=$((RESUBMIT - 1)) -J "m-$PROFILE" \
    -p "${SLURM_JOB_PARTITION:-gpu}" --gres="${GRES:-gpu:${SLURM_GPUS_ON_NODE:-1}}" \
    -c "${SLURM_CPUS_PER_TASK:-12}" --mem="${SLURM_MEM_PER_NODE:-65536}M" \
    isca/run_model.sbatch
fi

# Ollama is copied to the node's local disk first: loading its GPU libraries from the shared
# home storage can be slow enough that Ollama gives up on the GPU and silently runs on the CPU.
LOCAL=${TMPDIR:-/tmp}/ollama-$SLURM_JOB_ID
mkdir -p "$LOCAL" && cp -r "$PREFIX/ollama/." "$LOCAL/" && OLLAMA="$LOCAL/bin/ollama"
# The GPUs run with persistence mode off, so an idle GPU's driver state is unloaded and waking it
# can take longer than Ollama's 30-second GPU discovery watchdog; Ollama then falls back to the
# CPU. A small helper wakes the GPU first and keeps a CUDA context open for the whole job
# (about 0.5 GB of GPU memory), so discovery always finds a GPU that is already awake.
KEEPER=""
if [[ "${GRES:-}" == *1g.* ]]; then
  # MIG slice (10 GB): the slice needs no wake-up and has no room for extras, so UniXcoder runs on the CPU
  export MACS_EMBED_DEVICE=cpu
  echo "MIG slice: no GPU helper; UniXcoder on CPU"
else
  python - > "data/jobs/gpu-keeper-$SLURM_JOB_ID.log" 2>&1 <<'PY' &
import time, torch
t = time.time()
for i in range(torch.cuda.device_count()):
    torch.zeros(1, device=f"cuda:{i}")
print(f"GPU awake: {torch.cuda.device_count()} device(s) in {time.time() - t:.1f} s", flush=True)
while True:
    time.sleep(3600)
PY
  KEEPER=$!
  for _ in $(seq 120); do grep -q "GPU awake" "data/jobs/gpu-keeper-$SLURM_JOB_ID.log" 2>/dev/null && break; sleep 2; done
  cat "data/jobs/gpu-keeper-$SLURM_JOB_ID.log"
fi
export OLLAMA_VULKAN=false                   # CUDA only: do not spend discovery time on Vulkan
export OLLAMA_LLM_LIBRARY=cuda_v12           # one CUDA library (works with every A100 driver): one 30 s try, not three
SERVER=""
trap 'kill $SERVER $KEEPER 2>/dev/null || true; rm -rf "$LOCAL"' EXIT
# Ollama's GPU check has a fixed 30 s limit. On a busy node most of that goes on reading its large
# CUDA libraries, so they are read into memory right before each attempt, and the CUDA kernel cache
# is kept in a permanent folder instead of being rebuilt for every job.
export CUDA_CACHE_PATH="$PREFIX/cuda-cache" CUDA_CACHE_MAXSIZE=4294967296
mkdir -p "$CUDA_CACHE_PATH"
for attempt in $(seq 20); do
  cat "$LOCAL"/lib/ollama/*.so* "$LOCAL"/lib/ollama/cuda_v12/* > /dev/null 2>&1
  "$OLLAMA" serve > "data/jobs/ollama-$SLURM_JOB_ID.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 90); do curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null && break; sleep 2; done
  sleep 5
  if grep -q 'inference compute.*library=CUDA' "data/jobs/ollama-$SLURM_JOB_ID.log"; then
    echo "GPU found by Ollama (attempt $attempt)"; break
  fi
  echo "Ollama did not find the GPU (attempt $attempt); restarting it"
  kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null; SERVER=""; sleep 15
done
if [ -z "$SERVER" ]; then
  echo "Ollama could not use the GPU after 20 attempts; stopping instead of running on the CPU."
  nvidia-smi
  exit 1
fi

# download this profile's model if it is not there yet, then pin every revision in the config
MODEL=$(python isca/pin_revisions.py --profile-model "$PROFILE")
"$OLLAMA" list | awk 'NR > 1 {print $1}' | grep -qx "$MODEL" || "$OLLAMA" pull "$MODEL"
OLLAMA="$OLLAMA" python isca/pin_revisions.py
"$OLLAMA" list

# work units: (task, seed), the slow tier-3 tasks first so no worker is left with them at the end
TASK_IDS=$(python -c "import sys; sys.path.insert(0, 'scripts'); from build_reference import resolve_tasks; print(' '.join(sorted((t.id for t in resolve_tasks('$TASKS')), reverse=True)))")
SEED_LIST=$(python -c "import sys; sys.path.insert(0, 'scripts'); from run_experiment import parse_seeds; print(' '.join(map(str, parse_seeds('$SEEDS'))))")
Q=data/jobs/queue-$PROFILE-$SLURM_JOB_ID
mkdir -p "$Q"

worker() {
  local w=$1
  for t in $TASK_IDS; do
    for s in $SEED_LIST; do
      mkdir "$Q/$t-$s" 2>/dev/null || continue
      echo "[worker $w] $(date +%H:%M) $t seed $s"
      flock "$Q/ref-$t.lock" python scripts/build_reference.py --tasks "$t" --n 100 --profile "$PROFILE" --parallel "$WORKERS" >> "$Q/worker$w.log" 2>&1
      python scripts/run_experiment.py --tasks "$t" --conditions "$CONDITIONS" --seeds "$s" --profile "$PROFILE" >> "$Q/worker$w.log" 2>&1
    done
  done
}
if [ "$LEFT" != 0 ]; then
  for w in $(seq "$WORKERS"); do worker "$w" & sleep 5; done
  wait
  grep -h "FAILED" "$Q"/worker*.log | tail -20
  echo "== sessions left for $PROFILE: $(remaining) ($(date))"
fi

if [ "$SEARCH" = 1 ] && [ "$(remaining)" = 0 ] && [ ! -f "$DONE_SEARCH" ]; then
  echo "== open-problem search on $PROFILE ($(date))"
  python scripts/search_open.py run --tasks t3_circle_packing,t3_bin_packing,t3_cap_set \
    --budget 40 --seeds 1-3 --profile "$PROFILE" && touch "$DONE_SEARCH"
  python scripts/search_open.py report
fi

if [ "$(remaining)" = 0 ]; then
  tar -czf "data/jobs/results-$PROFILE.tgz" data/traces data/reference/models data/search data/jobs/*.log 2>/dev/null
  echo "== all done for $PROFILE. Download data/jobs/results-$PROFILE.tgz to the laptop."
fi
echo "== job $SLURM_JOB_ID finished $(date)"
