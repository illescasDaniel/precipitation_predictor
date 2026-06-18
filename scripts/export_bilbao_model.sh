#!/usr/bin/env bash
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/internal/run_python.sh" export_bilbao_model "$@"
