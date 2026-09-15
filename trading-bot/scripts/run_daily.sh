#!/bin/bash
# Runs one paper-trading decision via Alpaca and logs the result.
# Meant to be called by cron (see README.md "Running automatically").
#
# Reads credentials from .env in the project root — copy .env.example to
# .env and fill in your own Alpaca keys before using this. Never commit
# .env (it's gitignored).
#
# By default this stays on Alpaca's PAPER endpoint (no --live flag). Do
# not add --live to this script casually — going live should stay a
# deliberate, one-off decision, not something a cron job does silently.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

TICKER="${TRADING_BOT_TICKER:-AAPL}"
WEEKLY_BUDGET="${TRADING_BOT_WEEKLY_BUDGET:-50}"

mkdir -p logs
PYTHON="$SCRIPT_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$PYTHON" main.py live-trade --ticker "$TICKER" --weekly-budget "$WEEKLY_BUDGET"
  echo
} >> logs/live-trade.log 2>&1
