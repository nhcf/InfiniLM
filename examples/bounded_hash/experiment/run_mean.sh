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
mode="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  plan|run) exec "$CONTROLLER" mean_suite.py "$mode" "$@" ;;
  check) exec "$CONTROLLER" -m unittest -v test_affinity_router test_cache_telemetry test_gateway_integration test_matrix test_mean_router test_mean_suite ;;
  *) echo 'Usage: bash run_mean.sh {plan|check|run} [--profile prefix97|replay94] [--concurrency 16 32 64] [--seeds 33959 33960]' >&2; exit 2 ;;
esac
