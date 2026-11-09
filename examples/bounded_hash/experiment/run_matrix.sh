#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
readonly CONTROLLER='/mnt/zj-data/data/chengcp/infinilm/qwen3_32b_fourway_20261021/.controller/bin/python'
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
if [[ ! -x "$CONTROLLER" ]]; then
  echo "Existing controller missing: $CONTROLLER" >&2
  exit 2
fi
"$CONTROLLER" -c 'import sys, aiohttp; assert sys.prefix != sys.base_prefix'
echo 'NO_DOWNLOAD_NO_INSTALL_NO_REBUILD'
case "${1:-plan}" in
  plan|run) exec "$CONTROLLER" matrix.py "${1:-plan}" ;;
  check) exec "$CONTROLLER" -m unittest -v test_affinity_router test_cache_telemetry test_gateway_integration test_matrix ;;
  *) echo 'Usage: bash run_matrix.sh {plan|check|run}' >&2; exit 2 ;;
esac
