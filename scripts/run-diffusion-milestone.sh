#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run-diffusion-milestone.sh \
    --checkpoint-dir <path> \
    --data <dir-or-file> \
    --exp-id <id> \
    --add-steps <n> \
    [--backend cpu|gpu|cuda] \
    [--log-every <n>] \
    [--checkpoint-every <n>] \
    [--erm-bin <path>] \
    [--config <path>]

Description:
  Runs an additional diffusion-training segment and resumes from
  <checkpoint-dir>/latest when it exists.
USAGE
}

CHECKPOINT_DIR=""
DATA_PATH=""
EXP_ID=""
ADD_STEPS=""
BACKEND="cuda"
LOG_EVERY="25"
CHECKPOINT_EVERY="250"
ERM_BIN="./target/release/erm"
CONFIG_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --data) DATA_PATH="$2"; shift 2 ;;
    --exp-id) EXP_ID="$2"; shift 2 ;;
    --add-steps) ADD_STEPS="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
    --erm-bin) ERM_BIN="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT_DIR" || -z "$DATA_PATH" || -z "$EXP_ID" || -z "$ADD_STEPS" ]]; then
  echo "Missing required args." >&2
  usage
  exit 1
fi

if [[ ! -x "$ERM_BIN" ]]; then
  echo "erm binary not executable: $ERM_BIN" >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

RESUME_ARGS=()
LATEST_DIR="$CHECKPOINT_DIR/latest"
if [[ -d "$LATEST_DIR" ]]; then
  RESUME_ARGS+=(--resume "$LATEST_DIR")
fi

if [[ -z "$CONFIG_PATH" ]]; then
  if [[ -f "$LATEST_DIR/config.json" ]]; then
    CONFIG_PATH="$LATEST_DIR/config.json"
  elif [[ -f "$CHECKPOINT_DIR/final/config.json" ]]; then
    CONFIG_PATH="$CHECKPOINT_DIR/final/config.json"
  fi
fi

CONFIG_ARGS=()
if [[ -n "$CONFIG_PATH" ]]; then
  CONFIG_ARGS+=(--config "$CONFIG_PATH")
fi

echo "=== Diffusion Milestone Run ==="
echo "exp_id: $EXP_ID"
echo "data: $DATA_PATH"
echo "checkpoint_dir: $CHECKPOINT_DIR"
echo "add_steps: $ADD_STEPS"
echo "backend: $BACKEND"
echo "log_every: $LOG_EVERY"
echo "checkpoint_every: $CHECKPOINT_EVERY"
if [[ ${#RESUME_ARGS[@]} -gt 0 ]]; then
  echo "resume: ${RESUME_ARGS[*]}"
else
  echo "resume: (none)"
fi
if [[ ${#CONFIG_ARGS[@]} -gt 0 ]]; then
  echo "config: $CONFIG_PATH"
else
  echo "config: (default)"
fi

"$ERM_BIN" diffusion-train \
  --data "$DATA_PATH" \
  --steps "$ADD_STEPS" \
  --backend "$BACKEND" \
  --log-every "$LOG_EVERY" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --exp-id "$EXP_ID" \
  "${CONFIG_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
