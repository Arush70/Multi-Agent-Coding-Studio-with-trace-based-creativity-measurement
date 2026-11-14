#!/bin/bash
#SBATCH -A COMM514
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J m-rating
#SBATCH -o data/jobs/rating-%j.log
#SBATCH --export=ALL
#SBATCH -D .
#
# Rater materials and the AI rater, from the main model's traces:
#   1. build  - 60 pairs of correct solutions (Part A, SC4) and 48 labelling items (SC3)
#   2. export - rating booklet (HTML) and the Microsoft Forms Quick-import file (.docx), Part A
#               only, as in the approved ethics documents (v1.0); --with-part-b adds the items
#   3. judge  - the AI rater (DeepSeek-Coder-V2, a different family from the models studied) answers
#               the pairs and labels the 48 items. On the pairs it is reported separately from the
#               human raters; the 48 SC3 labels come only from AI raters (amendment 2026-11-14).
# The pairs are frozen once responses exist (build refuses to overwrite them).
#
#   sbatch isca/rating.sbatch
set -uo pipefail
PROFILE=${PROFILE:-deepseek}
MAIN=${MAIN:-qwen3.5:27b}
WORKERS=1
PREFIX=${MACS_PREFIX:-$HOME/macs-tools}
source "$PREFIX/miniforge3/etc/profile.d/conda.sh"
conda activate macs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_HOME="$PREFIX/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OLLAMA_MODELS="$PREFIX/models" OLLAMA_HOST="127.0.0.1:$((11500 + SLURM_JOB_ID % 400))"
export OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=16384
# never prune at start-up: a new server would otherwise delete a download that another job has in
# progress in the shared model folder, so a large model restarts from 0%
export OLLAMA_NOPRUNE=1
OLLAMA="$PREFIX/ollama/bin/ollama"
echo "== rating job $SLURM_JOB_ID on $(hostname) $(date)"
python -c "import docx" 2>/dev/null || pip install -q python-docx openpyxl

if [ ! -f data/rating/pairs.json ]; then
  python scripts/rating_study.py build --model "$MAIN" || exit 1
fi
python scripts/rating_study.py export

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
  # the server answers before GPU discovery has finished (two GPUs take longer than one), so wait
  # up to 90 s for discovery to report its result instead of checking after a fixed 5 s
  for _ in $(seq 45); do grep -qE 'inference compute|no compatible GPUs' "data/jobs/ollama-$SLURM_JOB_ID.log" && break; sleep 2; done
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
# a pull of a very large model can stop part-way (it resumes where it left off), so retry, and
# stop the job if the model is still missing: sessions without the model would all fail at once
for try in $(seq 10); do
  "$OLLAMA" list | awk 'NR > 1 {print $1}' | grep -qx "$MODEL" && break
  echo "pulling $MODEL (try $try)"; "$OLLAMA" pull "$MODEL" || sleep 60
done
if ! "$OLLAMA" list | awk 'NR > 1 {print $1}' | grep -qx "$MODEL"; then
  echo "model $MODEL could not be downloaded; stopping before any session runs"
  exit 1
fi
OLLAMA="$OLLAMA" python isca/pin_revisions.py
"$OLLAMA" list

python scripts/rating_study.py judge --profile "$PROFILE"
python scripts/rating_study.py analyse || true
tar -czf data/jobs/rating_results.tgz data/rating data/calibration 2>/dev/null
echo "== rating job finished $(date); download data/jobs/rating_results.tgz"
