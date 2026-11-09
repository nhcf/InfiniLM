#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
controller_python="${CONTROLLER_PYTHON:-/mnt/zj-data/data/chengcp/infinilm/qwen3_32b_fourway_20261021/.controller/bin/python}"
if [[ ! -x "$controller_python" ]]; then
  echo "控制端 Python 不存在：$controller_python。请把 CONTROLLER_PYTHON 设置为原实验使用的解释器。" >&2
  exit 2
fi
action="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  plan|check) exec "$controller_python" -B optimization_runner.py "$action" "$@" ;;
  selftest) exec "$controller_python" -B -m unittest discover -q -b "$@" ;;
  run)
    "$controller_python" -B -m unittest discover -q -b
    exec "$controller_python" -B optimization_runner.py run "$@"
    ;;
  *) echo '用法: bash run_optimized.sh plan|check|run，或 bash run_optimized.sh selftest' >&2; exit 2 ;;
esac
