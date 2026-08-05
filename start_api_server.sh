#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-7091}

exec python3 "$SCRIPT_DIR/local_motion_query_api.py" --host "$HOST" --port "$PORT" "$@"
