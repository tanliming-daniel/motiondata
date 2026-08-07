#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
FRP_DIR=${FRP_DIR:-/data5/cy/frp}
FRPC_BIN=${FRPC_BIN:-$FRP_DIR/frpc}
FRPC_CONFIG=${FRPC_CONFIG:-$PROJECT_ROOT/config/frpc.ini}
LOCAL_PORT=7091
REMOTE_PORT=7091
LOG_FILE=${LOG_FILE:-$PROJECT_ROOT/runtime/frpc.log}
PID_FILE=${PID_FILE:-$PROJECT_ROOT/runtime/frpc.pid}
RUNTIME_CONFIG=${RUNTIME_CONFIG:-$PROJECT_ROOT/runtime/frpc.ini}

mkdir -p "$PROJECT_ROOT/runtime"

if [ ! -x "$FRPC_BIN" ]; then
  echo "frpc not found or not executable: $FRPC_BIN" >&2
  exit 1
fi

if [ ! -f "$FRPC_CONFIG" ]; then
  echo "frpc config not found: $FRPC_CONFIG" >&2
  exit 1
fi

sed \
  -e "s/^local_port *= *.*/local_port = $LOCAL_PORT/" \
  -e "s/^remote_port *= *.*/remote_port = $REMOTE_PORT/" \
  "$FRPC_CONFIG" > "$RUNTIME_CONFIG"

nohup setsid "$FRPC_BIN" -c "$RUNTIME_CONFIG" > "$LOG_FILE" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"

echo "FRP client started"
echo "PID: $PID"
echo "Config template: $FRPC_CONFIG"
echo "Runtime config: $RUNTIME_CONFIG"
echo "Log: $LOG_FILE"
echo "Public endpoint: http://42.193.117.211:$REMOTE_PORT"
