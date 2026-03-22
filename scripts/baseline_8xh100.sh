#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
main() {
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

apply_env_overrides() {
  for assignment in "$@"; do
    if [[ "$assignment" != *=* ]]; then
      echo "Unexpected argument: $assignment" >&2
      echo "Pass overrides as KEY=VALUE, for example RUN_ID=myrun MAX_WALLCLOCK_SECONDS=60" >&2
      exit 2
    fi

    key="${assignment%%=*}"
    value="${assignment#*=}"
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "Invalid override name: $key" >&2
      exit 2
    fi
    export "$key=$value"
  done
}

reset_runtime_env() {
  unset RUN_ID
  unset SEED
  unset DATA_PATH
  unset TOKENIZER_PATH
  unset VOCAB_SIZE
  unset MAX_WALLCLOCK_SECONDS
  unset TRAIN_LOG_EVERY
  unset VAL_LOSS_EVERY
}

reset_runtime_env
apply_env_overrides "$@"

PYTHON="${PYTHON:-python3}"
RUN_ID="${RUN_ID:-baseline_8xh100_$(date +%Y%m%d_%H%M%S)}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python interpreter '$PYTHON' not found"
  exit 1
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found in PATH"
  exit 1
fi

if [ ! -f "./data/tokenizers/fineweb_1024_bpe.model" ] || [ ! -f "./data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin" ]; then
  echo "Downloading baseline dataset/tokenizer (sp1024, 80 train shards)..."
  "$PYTHON" data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export RUN_ID
export SEED="${SEED:-2000}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

echo "Running baseline-style training on 8xH100"
echo "RUN_ID=$RUN_ID"
echo "SEED=$SEED"
echo "DATA_PATH=$DATA_PATH"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"
echo "VOCAB_SIZE=$VOCAB_SIZE"
echo "MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"

torchrun --standalone --nproc_per_node=8 train_gpt.py
}

( main "$@" )
