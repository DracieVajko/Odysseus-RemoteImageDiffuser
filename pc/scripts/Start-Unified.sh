#!/usr/bin/env bash
# Start-Unified.sh — Mac/Linux launcher (venv + uvicorn). Lazy by default.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"
if [ ! -x "$VENV/bin/python" ] && [ -x "../.venv/bin/python" ]; then VENV="../.venv"; fi
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "Missing venv python: $PY" >&2
  echo "Create with: python3 -m venv $VENV && $VENV/bin/pip install -r requirements.txt" >&2
  exit 1
fi
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_VERBOSITY=warning
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PORT="${PORT:-8101}"
HOST="${HOST:-127.0.0.1}"
WHISPER_URL="${WHISPER_URL:-http://127.0.0.1:4444}"
if [ -n "${NO_ENGINE:-}" ]; then export UNIFIED_NO_ENGINE=1; fi
export UNIFIED_WHISPER_URL="$WHISPER_URL"
mkdir -p logs run
TS=$(date +%Y%m%d-%H%M%S)
LOG="logs/unified-${TS}.log"
echo "Starting unified server (LAZY): bind ${HOST}:${PORT}, whisper fallback ${WHISPER_URL}"
nohup "$PY" -m uvicorn unified_server:app --host "$HOST" --port "$PORT" >"$LOG" 2>&1 &
PID=$!
echo "$PID" > run/unified.pid
echo "PID $PID, waiting for /health ..."
for i in $(seq 1 120); do
  if ! kill -0 "$PID" 2>/dev/null; then echo "process exited early"; cat "$LOG"; exit 1; fi
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "READY: http://127.0.0.1:$PORT  WebUI: http://127.0.0.1:$PORT/webui"
    echo "Stop: kill \$(cat run/unified.pid)"
    exit 0
  fi
  sleep 5
done
echo "did not become ready; see $LOG" >&2
exit 1
