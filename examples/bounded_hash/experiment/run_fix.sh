#!/usr/bin/env bash
# Fresh-connection confirmation. Self-tests run before any benchmark stage.
set -euo pipefail
cd "$(dirname "$0")"
action="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  selftest|plan) exec bash run_next.sh "$action" "$@" ;;
  check) exec bash run_next.sh custom check --historical-run frozen_report "$@" ;;
  confirm|expand|expand-confirm)
    bash run_next.sh selftest
    exec bash run_next.sh "$action" "$@"
    ;;
  custom) exec bash run_next.sh custom "$@" ;;
  *) echo '用法: bash run_fix.sh [plan|selftest|check|confirm|expand|expand-confirm|custom]' >&2
     exit 2 ;;
esac
