#!/usr/bin/env bash
set -euo pipefail

# One-click setup for TSGB
# Usage: curl -fsSL <url>/oneclick_setup.sh | bash
# Optional env vars: REPO_URL, BRANCH, INSTALL_DIR, RUN_TRAIN (1 to train, 0 to skip), TRAIN_ARGS

REPO_URL="${REPO_URL:-https://gith.misile.xyz/h3.git:/projects/dsb/tsgb.git}"
BRANCH="${BRANCH:-main}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/tsgb}"
RUN_TRAIN="${RUN_TRAIN:-1}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

log() { echo "[$(date -Iseconds)] $*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1"; exit 1; }; }

need git
need curl

# Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
  log "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  log "uv already installed"
fi

# Clone or update repo
log "Using install dir: $INSTALL_DIR"
mkdir -p "$(dirname "$INSTALL_DIR")"
if [ -d "$INSTALL_DIR/.git" ]; then
  log "Repository exists, updating..."
  git -C "$INSTALL_DIR" fetch origin
  git -C "$INSTALL_DIR" checkout "$BRANCH"
  git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
else
  log "Cloning repository..."
  git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

log "Syncing dependencies with uv (honors uv.lock)..."
uv sync

# Prefer .env next to this script; otherwise fall back to example
SOURCE_ENV="$SCRIPT_DIR/.env"
if [ -f "$SOURCE_ENV" ]; then
  cp "$SOURCE_ENV" "$INSTALL_DIR/.env"
  log "Copied .env from script directory into repo"
elif [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  log "Created .env from .env.example (please edit secrets, e.g., VAST_API_KEY)"i

if [ "$RUN_TRAIN" != "0" ]; then
  log "Starting local training (this may take a while)..."
  uv run tsgb worker run $TRAIN_ARGS
  log "Training run finished. Check logs and checkpoints for results."
else
  log "Skipping training (set RUN_TRAIN=1 to enable)."
fi

log "Setup complete. Common next steps:"
echo "  - Edit .env to confirm VAST_API_KEY and storage settings"
echo "  - Check offers: uv run tsgb manager offers --limit 5"
echo "  - Start manager loop: uv run python -m tsgb.manager"
