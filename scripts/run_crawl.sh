#!/usr/bin/env bash
# Start (or resume) the snowball crawl in a detached tmux session, so it keeps
# running after you log off the VM. Resumes from any existing matches.csv.
# Usage:  bash scripts/run_crawl.sh [target]      (default target = 10000)

set -euo pipefail
TARGET="${1:-10000}"
cd "$(dirname "$0")/.."

tmux new-session -d -s crawl \
  ".venv/bin/python -u -m src.dataset \
     --seed='s1mple-_---' --seed='ZywOo' --seed='m0NESY' --seed='ropz' --seed='donk' \
     --seed='-ZeroSanity-' --seed='cigarette66' --seed='broky' --seed='b1t' --seed='sh1ro' \
     --target=$TARGET --per-player=30 --out=data/processed/matches.csv 2>&1 | tee -a crawl.log"

echo "Crawl started in tmux session 'crawl' (target=$TARGET)."
echo "  progress:  tail -f crawl.log     (or: wc -l data/processed/matches.csv)"
echo "  attach:    tmux attach -t crawl   (detach again with Ctrl-b then d)"
echo "  stop:      tmux kill-session -t crawl   (checkpointed; safe to resume later)"
