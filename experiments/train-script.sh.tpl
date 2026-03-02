set -eo pipefail
EXP_ID="__EXP_ID__"
echo "=== ERM Experiment: ${EXP_ID} ==="
echo "Host: $(hostname) | Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

apt-get update -qq 2>&1 | tail -1 || true
apt-get install -y -qq --no-install-recommends libvulkan1 mesa-vulkan-drivers ca-certificates 2>&1 | tail -3 || true
mkdir -p /etc/vulkan/icd.d
echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}' > /etc/vulkan/icd.d/nvidia_icd.json || true

BOOK_DIR="/workspace/rust-pcn/data/books"
CORPUS="/workspace/erm-rust/data/text/corpus_books.txt"
mkdir -p /workspace/erm-rust/data/text
if [ ! -f "${CORPUS}" ] || [ $(wc -c < "${CORPUS}" 2>/dev/null || echo 0) -lt 1000 ]; then
  ls "${BOOK_DIR}"/*.txt | head -80 | xargs cat > "${CORPUS}" 2>/dev/null || true
fi

EXP_DIR="/workspace/erm-rust/data/experiments/${EXP_ID}"
CKPT_DIR="${EXP_DIR}/checkpoints"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

cat > "${EXP_DIR}/config.json" <<'JSONEOF'
__CONFIG_JSON__
JSONEOF

echo "Config:"
cat "${EXP_DIR}/config.json"
echo ""
echo "=== Starting ${EXP_ID} training at $(date -u +%H:%M:%S) ==="
echo ""

/workspace/erm-rust/bin/erm colony-train \
  --data "${CORPUS}" \
  --steps 10000 \
  --config "${EXP_DIR}/config.json" \
  --backend gpu \
  --log-every 50 \
  --checkpoint-dir "${CKPT_DIR}" \
  2>&1 | tee "${LOG_DIR}/colony-train.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "=== ${EXP_ID} finished (exit=${EXIT_CODE}) at $(date -u +%H:%M:%S) ==="

if [ ${EXIT_CODE} -eq 0 ]; then
  WS="${EXP_DIR}/warmstart"
  mkdir -p "${WS}"
  cp "${CKPT_DIR}/graph.json" "${WS}/" 2>/dev/null || true
  cp "${CKPT_DIR}/ant_state.json" "${WS}/" 2>/dev/null || true
  cp "${EXP_DIR}/config.json" "${WS}/" 2>/dev/null || true
  echo "Warmstart saved to ${WS}"
fi
exit ${EXIT_CODE}
