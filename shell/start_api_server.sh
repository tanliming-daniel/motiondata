#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
HOST=${HOST:-0.0.0.0}
PORT=7091
SEMANTIC_MODEL=${SEMANTIC_MODEL:-/data5/cy/models/bge-m3}
SEMANTIC_INDEX=${SEMANTIC_INDEX:-$PROJECT_ROOT/dataset/semantic_index}
SEMANTIC_DEVICE=${SEMANTIC_DEVICE:-cuda:3}

if [ -f "$SEMANTIC_MODEL/config.json" ] && [ -f "$SEMANTIC_INDEX/manifest.json" ]; then
  exec python3 "$PROJECT_ROOT/local_motion_query_api.py" "$@" --host "$HOST" --port "$PORT" \
    --semantic-model "$SEMANTIC_MODEL" --semantic-index "$SEMANTIC_INDEX" \
    --semantic-device "$SEMANTIC_DEVICE"
fi

exec python3 "$PROJECT_ROOT/local_motion_query_api.py" "$@" --host "$HOST" --port "$PORT"
