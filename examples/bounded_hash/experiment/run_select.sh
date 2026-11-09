#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
controller_python="${CONTROLLER_PYTHON:-/mnt/zj-data/data/chengcp/infinilm/qwen3_32b_fourway_20261021/.controller/bin/python}"
if [[ ! -x "$controller_python" ]]; then
  echo "控制端 Python 不存在：$controller_python。请设置 CONTROLLER_PYTHON 为上次跑通的解释器。" >&2
  exit 2
fi
action="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  selftest) exec "$controller_python" -B -m unittest discover -q -b "$@" ;;
  plan|check) exec "$controller_python" -B select_version.py "$action" "$@" ;;
  run)
    "$controller_python" -B -m unittest discover -q -b
    exec "$controller_python" -B select_version.py run "$@"
    ;;
  *) echo '用法: bash run_select.sh [plan|selftest|check|run]' >&2; exit 2 ;;
esac
