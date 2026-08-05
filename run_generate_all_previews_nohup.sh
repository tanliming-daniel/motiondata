#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/nas/cy/humanmotion/multimotion_previews}
WIDTH=${WIDTH:-384}
HEIGHT=${HEIGHT:-240}
DATASET=${DATASET:-all}
PORT=${PORT:-7192}
CONVERTER_ENV=${CONVERTER_ENV:-sam2dam2}
CONDA_EXE=${CONDA_EXE:-conda}
LOG_FILE=${LOG_FILE:-$OUTPUT_ROOT/generate_all_previews.log}
PID_FILE=${PID_FILE:-$OUTPUT_ROOT/generate_all_previews.pid}

mkdir -p "$OUTPUT_ROOT"

if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE" || true)
  if [ -n "${OLD_PID:-}" ] && ps -p "$OLD_PID" >/dev/null 2>&1; then
    echo "Preview generation is already running: PID $OLD_PID"
    echo "Log: $LOG_FILE"
    exit 0
  fi
fi

cd "$SCRIPT_DIR"

nohup python generate_motion_previews.py \
  --output-root "$OUTPUT_ROOT" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --dataset "$DATASET" \
  --include-uncached \
  --converter-env "$CONVERTER_ENV" \
  --conda-exe "$CONDA_EXE" \
  --port "$PORT" \
  > "$LOG_FILE" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$PID_FILE"

echo "Started full multimotion preview generation"
echo "PID: $PID"
echo "Output: $OUTPUT_ROOT"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "Follow log: tail -f $LOG_FILE"
