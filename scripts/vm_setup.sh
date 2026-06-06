#!/usr/bin/env bash
# One-time setup for running the dataset crawler on a fresh Ubuntu VM
# (e.g. an Azure B1s). Installs system deps, clones the repo, builds the
# virtualenv, and installs Python requirements. After this runs, add your
# FACEIT API key to .env and start the crawl in tmux (see the guide).

set -euo pipefail

REPO_URL="https://github.com/samhussain25/cs2-match-predictor.git"
PROJECT_DIR="cs2-match-predictor"

echo ">>> Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git tmux

echo ">>> Cloning repo..."
if [ ! -d "$PROJECT_DIR" ]; then
    git clone "$REPO_URL"
fi
cd "$PROJECT_DIR"

echo ">>> Creating virtualenv and installing requirements..."
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Write .env automatically if the key was passed in the environment, e.g.:
#   FACEIT_API_KEY=xxxx bash vm_setup.sh
if [ -n "${FACEIT_API_KEY:-}" ]; then
    echo "FACEIT_API_KEY=$FACEIT_API_KEY" > .env
    echo ">>> Wrote .env from FACEIT_API_KEY."
fi

echo
echo "=================================================================="
echo "Setup complete."
if [ ! -f .env ]; then
    echo "  1) Add your FACEIT API key:"
    echo "       cd $PROJECT_DIR && echo 'FACEIT_API_KEY=PASTE_KEY' > .env"
fi
echo "  Start the crawl (detached tmux; survives logout):"
echo "       cd $PROJECT_DIR && bash scripts/run_crawl.sh 10000"
echo "  Watch:   tail -f $PROJECT_DIR/crawl.log"
echo "  When done, build Elo features then copy results back to your laptop:"
echo "       .venv/bin/python -m src.elo --data data/processed/matches.csv"
echo "=================================================================="
