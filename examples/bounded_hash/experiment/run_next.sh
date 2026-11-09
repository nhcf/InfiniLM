#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
controller_python="${CONTROLLER_PYTHON:-/mnt/zj-data/data/chengcp/infinilm/qwen3_32b_fourway_20261021/.controller/bin/python}"
if [[ ! -x "$controller_python" ]]; then
  echo "请设置 CONTROLLER_PYTHON 为上次成功测试使用的控制端 Python：$controller_python" >&2
  exit 2
fi
action="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  selftest) exec "$controller_python" -B -m unittest discover -q -b "$@" ;;
  plan) stage="${1:-confirm}"; if [[ $# -gt 0 ]]; then shift; fi
        exec "$controller_python" -B next_step.py plan "$stage" "$@" ;;
  confirm|expand|expand-confirm)
        exec "$controller_python" -B next_step.py run "$action" "$@" ;;
  custom) exec "$controller_python" -B cache_confirm.py "$@" ;;
  *) echo '用法: bash run_next.sh selftest | plan [confirm|expand|expand-confirm] | confirm | expand | expand-confirm | custom ...' >&2
     exit 2 ;;
esac
