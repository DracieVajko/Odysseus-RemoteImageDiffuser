#!/usr/bin/env bash
# Stop-Unified.sh
set -u
cd "$(dirname "$0")"
if [ -f run/unified.pid ]; then
  PID=$(cat run/unified.pid)
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping unified PID $PID ..."
    kill "$PID" 2>/dev/null
    sleep 3
  fi
  rm -f run/unified.pid
fi
echo "Done."
