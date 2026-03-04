#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run-reasoning-resume.sh \
    --checkpoint-dir <path> \
    --exp-id <id> \
    [--data <dir>] \
    [--add-steps <n>] \
    [--backend cpu|gpu|cuda] \
    [--erm-bin <path>] \
    [--config <path>] \
    [--log-every <n>] \
    [--checkpoint-every <n>]

Description:
  Resume diffusion training on a reasoning Q/A corpus prepared by
  scripts/prepare-reasoning-corpus.py. This keeps checkpoint continuity
  while switching data domain.
USAGE
}

CHECKPOINT_DIR=""
EXP_ID=""
DATA_PATH="/workspace/rust-pcn/data/reasoning-qa-sharded"
ADD_STEPS="200000"
BACKEND="cuda"
ERM_BIN="/workspace/erm-rust/bin/erm.new"
CONFIG_PATH=""
LOG_EVERY="25"
CHECKPOINT_EVERY="250"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --exp-id) EXP_ID="$2"; shift 2 ;;
    --data) DATA_PATH="$2"; shift 2 ;;
    --add-steps) ADD_STEPS="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --erm-bin) ERM_BIN="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${CHECKPOINT_DIR}" || -z "${EXP_ID}" ]]; then
  echo "Missing required args: --checkpoint-dir and --exp-id" >&2
  usage
  exit 1
fi

if [[ ! -x "${ERM_BIN}" ]]; then
  echo "erm binary not executable: ${ERM_BIN}" >&2
  exit 1
fi

if [[ ! -d "${DATA_PATH}" ]]; then
  echo "reasoning data dir not found: ${DATA_PATH}" >&2
  exit 1
fi

if [[ ! -d "${CHECKPOINT_DIR}/latest" ]]; then
  echo "missing latest checkpoint dir at: ${CHECKPOINT_DIR}/latest" >&2
  exit 1
fi

echo "=== Reasoning Resume Run ==="
echo "exp_id: ${EXP_ID}"
echo "data: ${DATA_PATH}"
echo "checkpoint_dir: ${CHECKPOINT_DIR}"
echo "add_steps: ${ADD_STEPS}"
echo "backend: ${BACKEND}"
echo "log_every: ${LOG_EVERY}"
echo "checkpoint_every: ${CHECKPOINT_EVERY}"

CMD=(
  "${ERM_BIN}" diffusion-train
  --data "${DATA_PATH}"
  --steps "${ADD_STEPS}"
  --backend "${BACKEND}"
  --log-every "${LOG_EVERY}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --exp-id "${EXP_ID}"
  --resume "${CHECKPOINT_DIR}/latest"
)

if [[ -n "${CONFIG_PATH}" ]]; then
  CMD+=(--config "${CONFIG_PATH}")
fi

printf 'command: %q ' "${CMD[@]}"; echo
"${CMD[@]}"
