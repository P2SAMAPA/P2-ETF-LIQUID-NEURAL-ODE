#!/usr/bin/env bash
# run_engine.sh — Entrypoint for P2-ETF-LIQUID-NEURAL-ODE engine
set -euo pipefail

UNIVERSE="${1:-combined}"
MODE="${2:-train}"
CONFIG="${3:-ltc_config.toml}"

echo "=========================================="
echo " P2Quant · LIQUID-NEURAL-ODE Engine"
echo " Universe : $UNIVERSE"
echo " Mode     : $MODE"
echo " Config   : $CONFIG"
echo "=========================================="

# Activate venv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated .venv"
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python main.py --universe "$UNIVERSE" --mode "$MODE" --config "$CONFIG"
