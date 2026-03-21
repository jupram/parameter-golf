#!/usr/bin/env bash
set -eEuo pipefail

on_error() {
  local exit_code=$?
  local line_no=$1
  local command=$2
  echo "ERROR: command failed at line ${line_no}: ${command}" >&2
  echo "Exit code: ${exit_code}" >&2
  if [[ -t 0 ]]; then
    read -r -p "Press Enter to exit..." _
  fi
  exit "${exit_code}"
}

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
  unset WARMUP_STEPS
  unset TRAIN_BATCH_TOKENS
  unset TRAIN_SEQ_LEN
  unset TRAIN_LOG_EVERY
  unset VAL_LOSS_EVERY
  unset VAL_MAX_TOKENS
  unset ENABLE_COMPILE
  unset TIE_EMBEDDINGS
  unset SKIP_FINAL_ROUNDTRIP_EVAL || true
}

reset_runtime_env
apply_env_overrides "$@"

PYTHON="${PYTHON:-./.venv/Scripts/python.exe}"
RUN_ID="${RUN_ID:-baseline_win5090_$(date +%Y%m%d_%H%M%S)}"

if [ ! -x "$PYTHON" ]; then
  echo "Python not found at $PYTHON"
  echo "Set PYTHON=... to your interpreter, for example ./venv/bin/python on Linux/WSL."
  exit 1
fi

if [ ! -f "./data/tokenizers/fineweb_1024_bpe.model" ] || [ ! -f "./data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin" ]; then
  echo "Downloading baseline dataset/tokenizer (sp1024, 80 train shards)..."
  "$PYTHON" data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
fi

export RUN_ID
export SEED="${SEED:-42}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export WARMUP_STEPS="${WARMUP_STEPS:-3}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-262144}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-262144}"
export ENABLE_COMPILE="${ENABLE_COMPILE:-0}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-0}"

echo "Running local 1xRTX baseline-style training"
echo "RUN_ID=$RUN_ID"
echo "SEED=$SEED"
echo "DATA_PATH=$DATA_PATH"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"
echo "VOCAB_SIZE=$VOCAB_SIZE"
echo "MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
echo "WARMUP_STEPS=$WARMUP_STEPS"
echo "TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS"
echo "TRAIN_SEQ_LEN=$TRAIN_SEQ_LEN"
echo "TRAIN_LOG_EVERY=$TRAIN_LOG_EVERY"
echo "VAL_LOSS_EVERY=$VAL_LOSS_EVERY"
echo "VAL_MAX_TOKENS=$VAL_MAX_TOKENS"
echo "ENABLE_COMPILE=$ENABLE_COMPILE"
echo "TIE_EMBEDDINGS=$TIE_EMBEDDINGS"
"$PYTHON" train_gpt.py
}

trap 'on_error "${LINENO}" "${BASH_COMMAND}"' ERR

main "$@"
