#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/nas/cy/humanmotion/multimotion_previews}
WIDTH=${WIDTH:-384}
HEIGHT=${HEIGHT:-240}
QUALITY=${QUALITY:-80}
WORKERS=${WORKERS:-8}
SHARD_COUNT=${SHARD_COUNT:-4}
DATASET=${DATASET:-all}
PORT=${PORT:-7192}
CONVERTER_ENV=${CONVERTER_ENV:-sam2dam2}
CONDA_EXE=${CONDA_EXE:-conda}
PYTHON_BIN=${PYTHON_BIN:-/data1/cy/anaconda3/bin/python}
LOG_FILE=${LOG_FILE:-$SCRIPT_DIR/generate_all_previews.log}
PID_FILE=${PID_FILE:-$SCRIPT_DIR/generate_all_previews.pid}

mkdir -p "$OUTPUT_ROOT"

if [ -f "$PID_FILE" ]; then
  while read -r OLD_PID; do
    if [ -n "${OLD_PID:-}" ] && ps -p "$OLD_PID" >/dev/null 2>&1; then
      echo "Preview generation is already running: PID $OLD_PID"
      exit 0
    fi
  done < "$PID_FILE"
fi

cd "$SCRIPT_DIR"

: > "$PID_FILE"
for ((SHARD_INDEX=0; SHARD_INDEX<SHARD_COUNT; SHARD_INDEX++)); do
  SHARD_PORT=$((PORT + SHARD_INDEX))
  SHARD_LOG="${LOG_FILE%.log}.shard-${SHARD_INDEX}.log"
  nohup setsid "$PYTHON_BIN" generate_motion_previews.py \
    --output-root "$OUTPUT_ROOT" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --quality "$QUALITY" \
    --workers "$WORKERS" \
    --skip-migration \
    --dataset "$DATASET" \
    --converter-env "$CONVERTER_ENV" \
    --conda-exe "$CONDA_EXE" \
    --port "$SHARD_PORT" \
    --shard-count "$SHARD_COUNT" \
    --shard-index "$SHARD_INDEX" \
    > "$SHARD_LOG" 2>&1 &
  PID=$!
  printf '%s\n' "$PID" >> "$PID_FILE"
  echo "Shard $SHARD_INDEX: PID $PID, port $SHARD_PORT, log $SHARD_LOG"
done

echo "Started full multimotion preview generation"
echo "Workers: $SHARD_COUNT"
echo "Output: $OUTPUT_ROOT"
echo "PID file: $PID_FILE"
echo "Follow logs: tail -f ${LOG_FILE%.log}.shard-*.log"
