#!/bin/bash
# One-time setup on ISCA. Run it from the repository folder inside an interactive job on a
# compute node (the ISCA docs ask that nothing is installed on the login nodes):
#
#   srun --export ALL -N 1 -t 02:00:00 -p sq -A COMM514 --mem 16G -c 4 --pty -- bash
#   bash isca/setup.sh
#
# It installs Python (Miniforge) and Ollama into $MACS_PREFIX (default ~/macs-tools), downloads
# the models in $MACS_MODELS (default qwen3.5:27b, about 17 GB) and UniXcoder, and writes
# config/models.yaml with the pinned revisions. Safe to run again: finished parts are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PREFIX=${MACS_PREFIX:-$HOME/macs-tools}
MODELS=${MACS_MODELS:-qwen3.5:27b}
mkdir -p "$PREFIX" data/jobs
echo "== tools go to $PREFIX (free space below; the 27B model needs about 17 GB) =="
df -h "$PREFIX" | tail -1 || true

echo "== 1/5 Python (Miniforge) =="
if [ ! -x "$PREFIX/miniforge3/bin/conda" ]; then
  curl -fsSL -o "$PREFIX/miniforge.sh" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$PREFIX/miniforge.sh" -b -p "$PREFIX/miniforge3"
  rm -f "$PREFIX/miniforge.sh"
fi
source "$PREFIX/miniforge3/etc/profile.d/conda.sh"
conda env list | grep -q "^macs " || conda create -y -n macs python=3.12 zstd
conda activate macs
pip install -q -e ".[dev,embed]" python-docx openpyxl   # torch from PyPI includes CUDA for the A100

echo "== 2/5 Ollama =="
if [ ! -x "$PREFIX/ollama/bin/ollama" ]; then
  mkdir -p "$PREFIX/ollama"
  if curl -fsSL -o "$PREFIX/ollama.tgz" https://ollama.com/download/ollama-linux-amd64.tgz; then
    tar -xzf "$PREFIX/ollama.tgz" -C "$PREFIX/ollama"
  else
    curl -fsSL -o "$PREFIX/ollama.tar.zst" https://ollama.com/download/ollama-linux-amd64.tar.zst
    tar --use-compress-program=unzstd -xf "$PREFIX/ollama.tar.zst" -C "$PREFIX/ollama"
  fi
  rm -f "$PREFIX"/ollama.tgz "$PREFIX"/ollama.tar.zst
fi
OLLAMA="$PREFIX/ollama/bin/ollama"
"$OLLAMA" --version || true

echo "== 3/5 models: $MODELS =="
export OLLAMA_MODELS="$PREFIX/models" OLLAMA_HOST=127.0.0.1:11555
"$OLLAMA" serve > "$PREFIX/ollama-setup.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null && break; sleep 2; done
for m in $MODELS; do "$OLLAMA" pull "$m"; done
"$OLLAMA" list | tee "$PREFIX/ollama-list.txt"

echo "== 4/5 UniXcoder (Facilitator) =="
export HF_HOME="$PREFIX/hf"
python -c "from transformers import AutoModel, AutoTokenizer; n='microsoft/unixcoder-base'; AutoTokenizer.from_pretrained(n); AutoModel.from_pretrained(n); print('UniXcoder ready')"

echo "== 5/5 config/models.yaml with pinned revisions =="
OLLAMA="$OLLAMA" python isca/pin_revisions.py

python -m pytest -q -x || echo "(some tests failed; send the output to Claude before submitting jobs)"
echo
echo "Setup done. Leave this job (type: exit), then from the login node run:"
echo "    sbatch isca/run_model.sbatch"
