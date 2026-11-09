#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
controller_python="${CONTROLLER_PYTHON:-/mnt/zj-data/data/chengcp/infinilm/qwen3_32b_fourway_20261021/.controller/bin/python}"
if [[ ! -x "$controller_python" ]]; then
  echo "Verified controller not found: $controller_python" >&2
  echo 'Set CONTROLLER_PYTHON to the Python used by your previously working benchmark. No environment was modified.' >&2
  exit 2
fi
if [[ "${1:-}" == selftest ]]; then
  shift
  exec "$controller_python" -B -m unittest discover -q "$@"
fi
exec "$controller_python" -B cache_suite.py "$@"
