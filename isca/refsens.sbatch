#!/bin/bash
#SBATCH -A COMM514
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J m-refsens
#SBATCH -o data/jobs/refsens-%j.log
#SBATCH --export=ALL
#SBATCH -D .
#
# Reference-size sensitivity (proposal 3.6: novelty checked with 50 and 200 reference samples).
# Tops up the main model's reference sets to 200 on six tasks fixed in advance (two per tier, the
# first two task ids in each tier) and compares novelty rankings at 50, 100 and 200. The main
# analysis keeps using the first 100. Seeds are fixed, so samples 0-99 are unchanged.
#
#   sbatch isca/refsens.sbatch
set -uo pipefail
PROFILE=${PROFILE:-qwen35_27b}
MODEL_NAME=${MODEL_NAME:-qwen3.5:27b}
TASKS=${TASKS:-t1_mbpp119_search,t1_mbpp130_max_occurrences,t2_humaneval0_has_close_elements,t2_humaneval20_find_closest_elements,t3_bin_packing,t3_cap_set}
WORKERS=6
PREFIX=${MACS_PREFIX:-$HOME/macs-tools}
source "$PREFIX/miniforge3/etc/profile.d/conda.sh"
conda activate macs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_HOME="$PREFIX/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OLLAMA_MODELS="$PREFIX/models" OLLAMA_HOST="127.0.0.1:$((11500 + SLURM_JOB_ID % 400))"
export OLLAMA_NUM_PARALLEL=$WORKERS OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=16384
OLLAMA="$PREFIX/ollama/bin/ollama"
echo "== reference sensitivity job $SLURM_JOB_ID on $(hostname) $(date)"

# Ollama is copied to the node's local disk first: loading its GPU libraries from the shared
# home storage can be slow enough that Ollama gives up on the GPU and silently runs on the CPU.
LOCAL=${TMPDIR:-/tmp}/ollama-$SLURM_JOB_ID
mkdir -p "$LOCAL" && cp -r "$PREFIX/ollama/." "$LOCAL/" && OLLAMA="$LOCAL/bin/ollama"
# The GPUs run with persistence mode off, so an idle GPU's driver state is unloaded and waking it
# can take longer than Ollama's 30-second GPU discovery watchdog; Ollama then falls back to the
# CPU. A small helper wakes the GPU first and keeps a CUDA context open for the whole job
# (about 0.5 GB of GPU memory), so discovery always finds a GPU that is already awake.
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
export OLLAMA_VULKAN=false                   # CUDA only: do not spend discovery time on Vulkan
export OLLAMA_LLM_LIBRARY=cuda_v12           # one CUDA library (works with every A100 driver): one 30 s try, not three
SERVER=""
trap 'kill $SERVER $KEEPER 2>/dev/null || true; rm -rf "$LOCAL"' EXIT
for attempt in $(seq 10); do
  "$OLLAMA" serve > "data/jobs/ollama-$SLURM_JOB_ID.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 90); do curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null && break; sleep 2; done
  sleep 5
  if grep -q 'inference compute.*library=CUDA' "data/jobs/ollama-$SLURM_JOB_ID.log"; then
    echo "GPU found by Ollama (attempt $attempt)"; break
  fi
  echo "Ollama did not find the GPU (attempt $attempt); restarting it"
  kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null; SERVER=""; sleep 30
done
if [ -z "$SERVER" ]; then
  echo "Ollama could not use the GPU after 10 attempts; stopping instead of running on the CPU."
  nvidia-smi
  exit 1
fi

# download this profile's model if it is not there yet, then pin every revision in the config
MODEL=$(python isca/pin_revisions.py --profile-model "$PROFILE")
"$OLLAMA" list | awk 'NR > 1 {print $1}' | grep -qx "$MODEL" || "$OLLAMA" pull "$MODEL"
OLLAMA="$OLLAMA" python isca/pin_revisions.py
"$OLLAMA" list

python scripts/build_reference.py --tasks "$TASKS" --n 200 --profile "$PROFILE" --parallel "$WORKERS"
python scripts/reference_sensitivity.py --model "$MODEL_NAME"
tar -czf data/jobs/refsens_results.tgz data/results/reference_sensitivity.json data/reference/models
echo "== reference sensitivity finished $(date); download data/jobs/refsens_results.tgz"
