#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# test-curriculum.sh — Test the curriculum supervisor's plateau detection,
#                      startup grace period, universal BPE vocab, and symlink
#
# Tests:
#   1-8:  Plateau detection unit tests
#   9:    End-to-end (startup grace + plateau + symlink)
#   10:   Metrics.jsonl symlink
#   11:   Startup timeout (no metrics)
#   12:   No misleading log messages during startup
#   13:   Universal BPE vocab building
#   14:   Pre-existing vocab is reused (not rebuilt)
#
# Usage: ./test-curriculum.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="/tmp/curriculum-test-$$"
PASS=0
FAIL=0

cleanup() {
  rm -rf "$TEST_DIR"
}
trap cleanup EXIT

mkdir -p "$TEST_DIR"

# ── Colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓ PASS${NC}: $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}✗ FAIL${NC}: $1"; FAIL=$((FAIL + 1)); }

# ──────────────────────────────────────────────────────────────────────────────
# Helper: plateau detection (isolated for unit tests)
# ──────────────────────────────────────────────────────────────────────────────

detect_plateau_test() {
  local metrics_file="$1"
  local window_secs="$2"
  local threshold="$3"

  if [[ ! -f "$metrics_file" ]]; then
    echo "0.0"
    return 1
  fi

  python3 -c "
import json, sys

metrics_file = '$metrics_file'
threshold = $threshold

records = []
try:
    with open(metrics_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                records.append(r)
            except json.JSONDecodeError:
                continue
except Exception:
    print('0.0')
    sys.exit(1)

if len(records) < 2:
    loss = records[-1]['loss'] if records else 0.0
    print(f'{loss:.6f}')
    sys.exit(1)

recent_count = max(4, min(len(records), 20))
recent = records[-recent_count:]

oldest_loss = recent[0]['loss']
newest_loss = recent[-1]['loss']
min_loss_in_window = min(r['loss'] for r in recent)

if oldest_loss <= 0:
    print(f'{newest_loss:.6f}')
    sys.exit(1)

improvement = (oldest_loss - min_loss_in_window) / abs(oldest_loss)

print(f'{newest_loss:.6f}')
if improvement < threshold:
    sys.exit(0)  # plateaued
else:
    sys.exit(1)  # still improving
" 2>/dev/null
}

# ──────────────────────────────────────────────────────────────────────────────
echo "=== Test 1: Flat loss → plateau detected ==="
# ──────────────────────────────────────────────────────────────────────────────

METRICS="$TEST_DIR/test1_metrics.jsonl"
for i in $(seq 1 20); do
  step=$((i * 50))
  echo "{\"exp_id\":\"test\",\"step\":$step,\"loss\":3.5,\"edits\":10,\"mean_phi\":0.1,\"deaths\":2,\"seq_len\":512,\"batch\":2,\"hidden_dim\":192}" >> "$METRICS"
done

set +e; detect_plateau_test "$METRICS" 300 0.005 > /dev/null 2>&1; RESULT=$?; set -e
if [[ $RESULT -eq 0 ]]; then
  pass "Flat loss (3.5 constant) detected as plateau"
else
  fail "Flat loss NOT detected as plateau (exit=$RESULT)"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 2: Improving loss → no plateau ==="
# ──────────────────────────────────────────────────────────────────────────────

METRICS="$TEST_DIR/test2_metrics.jsonl"
for i in $(seq 1 20); do
  step=$((i * 50))
  loss=$(python3 -c "print(f'{5.0 - 0.1 * $i:.4f}')")
  echo "{\"exp_id\":\"test\",\"step\":$step,\"loss\":$loss,\"edits\":10,\"mean_phi\":0.1,\"deaths\":2,\"seq_len\":512,\"batch\":2,\"hidden_dim\":192}" >> "$METRICS"
done

set +e; LOSS=$(detect_plateau_test "$METRICS" 300 0.005); RESULT=$?; set -e
if [[ $RESULT -eq 1 ]]; then
  pass "Improving loss (5.0→3.0) correctly NOT a plateau"
else
  fail "Improving loss was incorrectly flagged as plateau (exit=$RESULT, loss=$LOSS)"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 3: Tiny improvement (below threshold) → plateau ==="
# ──────────────────────────────────────────────────────────────────────────────

METRICS="$TEST_DIR/test3_metrics.jsonl"
for i in $(seq 1 20); do
  step=$((i * 50))
  loss=$(python3 -c "print(f'{3.5 - 0.0001 * $i:.6f}')")
  echo "{\"exp_id\":\"test\",\"step\":$step,\"loss\":$loss,\"edits\":10,\"mean_phi\":0.1,\"deaths\":2,\"seq_len\":512,\"batch\":2,\"hidden_dim\":192}" >> "$METRICS"
done

set +e; LOSS=$(detect_plateau_test "$METRICS" 300 0.005); RESULT=$?; set -e
if [[ $RESULT -eq 0 ]]; then
  pass "Tiny improvement (0.057%) correctly detected as plateau"
else
  fail "Tiny improvement NOT detected as plateau (exit=$RESULT, loss=$LOSS)"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 4: Just above threshold → NOT plateau ==="
# ──────────────────────────────────────────────────────────────────────────────

METRICS="$TEST_DIR/test4_metrics.jsonl"
for i in $(seq 1 20); do
  step=$((i * 50))
  loss=$(python3 -c "print(f'{3.5 - 0.00175 * $i:.6f}')")
  echo "{\"exp_id\":\"test\",\"step\":$step,\"loss\":$loss,\"edits\":10,\"mean_phi\":0.1,\"deaths\":2,\"seq_len\":512,\"batch\":2,\"hidden_dim\":192}" >> "$METRICS"
done

set +e; LOSS=$(detect_plateau_test "$METRICS" 300 0.005); RESULT=$?; set -e
if [[ $RESULT -eq 1 ]]; then
  pass "Above-threshold improvement (1.0%) correctly NOT a plateau"
else
  fail "Above-threshold improvement incorrectly flagged as plateau (exit=$RESULT, loss=$LOSS)"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 5: Empty metrics file → no crash, not plateau ==="
# ──────────────────────────────────────────────────────────────────────────────

METRICS="$TEST_DIR/test5_metrics.jsonl"
touch "$METRICS"

set +e; LOSS=$(detect_plateau_test "$METRICS" 300 0.005); RESULT=$?; set -e
if [[ $RESULT -eq 1 ]]; then
  pass "Empty metrics file: no crash, not plateau"
else
  fail "Empty metrics file incorrectly flagged as plateau"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 6: Missing metrics file → no crash, not plateau ==="
# ──────────────────────────────────────────────────────────────────────────────

set +e; LOSS=$(detect_plateau_test "$TEST_DIR/nonexistent.jsonl" 300 0.005); RESULT=$?; set -e
if [[ $RESULT -eq 1 ]]; then
  pass "Missing metrics file: no crash, not plateau"
else
  fail "Missing metrics file incorrectly flagged as plateau"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 7: Single record → not plateau (insufficient data) ==="
# ──────────────────────────────────────────────────────────────────────────────

METRICS="$TEST_DIR/test7_metrics.jsonl"
echo '{"exp_id":"test","step":50,"loss":3.5,"edits":10,"mean_phi":0.1,"deaths":2,"seq_len":512,"batch":2,"hidden_dim":192}' > "$METRICS"

set +e; LOSS=$(detect_plateau_test "$METRICS" 300 0.005); RESULT=$?; set -e
if [[ $RESULT -eq 1 ]]; then
  pass "Single record: not plateau (insufficient data)"
else
  fail "Single record incorrectly flagged as plateau"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 8: Noisy but improving → NOT plateau ==="
# ──────────────────────────────────────────────────────────────────────────────

METRICS="$TEST_DIR/test8_metrics.jsonl"
for i in $(seq 1 20); do
  step=$((i * 50))
  loss=$(python3 -c "
import math
base = 4.0 - 0.025 * $i
noise = 0.05 * math.sin($i * 2.3)
print(f'{base + noise:.6f}')
")
  echo "{\"exp_id\":\"test\",\"step\":$step,\"loss\":$loss,\"edits\":10,\"mean_phi\":0.1,\"deaths\":2,\"seq_len\":512,\"batch\":2,\"hidden_dim\":192}" >> "$METRICS"
done

set +e; LOSS=$(detect_plateau_test "$METRICS" 300 0.005); RESULT=$?; set -e
if [[ $RESULT -eq 1 ]]; then
  pass "Noisy but improving (4.0→3.5 with noise) correctly NOT a plateau"
else
  fail "Noisy improving loss incorrectly flagged as plateau (exit=$RESULT, loss=$LOSS)"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 9: End-to-end (startup grace + vocab + plateau) ==="
# ──────────────────────────────────────────────────────────────────────────────

# Create mock books
MOCK_BOOKS="$TEST_DIR/books"
mkdir -p "$MOCK_BOOKS"
echo "This is book one about Alice in Wonderland. The quick brown fox jumps over the lazy dog." > "$MOCK_BOOKS/book1.txt"
echo "This is book two about Through the Looking Glass. Pack my box with five dozen liquor jugs." > "$MOCK_BOOKS/book2.txt"
echo "This is book three about The Jabberwocky. How vexingly quick daft zebras jump." > "$MOCK_BOOKS/book3.txt"

# Create a mock config (minimal)
MOCK_CONFIG="$TEST_DIR/mock_config.json"
cat > "$MOCK_CONFIG" << 'CFGEOF'
{
  "vocab_size": 0,
  "seq_len": 64,
  "hidden_dim": 64,
  "tokenizer_type": "bpe",
  "bpe_vocab_size": 100,
  "bpe_vocab_path": ""
}
CFGEOF

# Create mock erm binary with startup delay
MOCK_BIN="$TEST_DIR/mock-erm"
cat > "$MOCK_BIN" << 'MOCKEOF'
#!/usr/bin/env bash
# Mock erm: 3s startup delay (simulating CUDA JIT), then flat metrics

CKPT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir) CKPT_DIR="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [[ -z "$CKPT_DIR" ]]; then
  echo "Mock erm: no --checkpoint-dir, exiting"
  exit 1
fi

METRICS_FILE="$CKPT_DIR/metrics.jsonl"
mkdir -p "$CKPT_DIR"

STEP=0
TRAP_CAUGHT=0

handle_term() {
  echo "Mock erm: SIGTERM caught, saving checkpoint..."
  echo "{\"checkpoint\":\"saved\",\"step\":$STEP}" > "$CKPT_DIR/checkpoint_saved.json"
  TRAP_CAUGHT=1
  exit 0
}
trap handle_term SIGTERM

# Phase 1: Startup delay (CUDA JIT simulation — no BPE since vocab is pre-built)
echo "Mock erm: startup phase (3s delay, no metrics)..."
sleep 3

# Phase 2: Write flat-loss metrics
for i in $(seq 1 10); do
  STEP=$((i * 50))
  echo "{\"exp_id\":\"test\",\"step\":$STEP,\"loss\":3.500,\"edits\":10,\"mean_phi\":0.1,\"deaths\":2,\"seq_len\":512,\"batch\":2,\"hidden_dim\":192}" >> "$METRICS_FILE"
done

echo "Mock erm: metrics written, waiting for SIGTERM..."
while [[ $TRAP_CAUGHT -eq 0 ]]; do
  sleep 0.5
done
exit 0
MOCKEOF
chmod +x "$MOCK_BIN"

# Run supervisor
MOCK_EXP="$TEST_DIR/experiment"
mkdir -p "$MOCK_EXP"

echo "[test9] Running supervisor with startup grace, vocab building, short timeouts..."

(
  "$SCRIPT_DIR/curriculum-supervisor.sh" \
    --books-dir "$MOCK_BOOKS" \
    --experiment-dir "$MOCK_EXP" \
    --config "$MOCK_CONFIG" \
    --erm-bin "$MOCK_BIN" \
    --backend cpu \
    --steps-per-book 10000 \
    --log-every 50 \
    --checkpoint-every 250 \
    --startup-timeout 30 \
    --plateau-window 3 \
    --plateau-threshold 0.005 \
    --hard-timeout 30 \
    --poll-interval 2 \
    --max-books 2 \
    --exp-id-prefix "test" \
    --bpe-num-merges 50 \
    --bpe-sample-mb 1
) > "$TEST_DIR/supervisor_output.log" 2>&1 &
SUP_PID=$!

WAITED=0
while kill -0 "$SUP_PID" 2>/dev/null && [[ $WAITED -lt 120 ]]; do
  sleep 2
  WAITED=$((WAITED + 2))
done
if kill -0 "$SUP_PID" 2>/dev/null; then
  echo "[test9] Supervisor still running after 120s, killing..."
  kill -9 "$SUP_PID" 2>/dev/null || true
fi
wait "$SUP_PID" 2>/dev/null || true

# Check results
if [[ -f "$MOCK_EXP/curriculum_state.json" ]]; then
  STATE_INDEX=$(python3 -c "
import json
with open('$MOCK_EXP/curriculum_state.json') as f:
  s = json.load(f)
print(s.get('book_index', -1))
")
  STATE_REASON=$(python3 -c "
import json
with open('$MOCK_EXP/curriculum_state.json') as f:
  s = json.load(f)
print(s.get('switch_reason', 'unknown'))
")

  if [[ $STATE_INDEX -ge 1 ]]; then
    pass "E2E: supervisor advanced through at least 2 books (index=$STATE_INDEX)"
  else
    fail "E2E: supervisor didn't advance (index=$STATE_INDEX)"
  fi

  if [[ "$STATE_REASON" == "plateau_detected" || "$STATE_REASON" == "hard_timeout" ]]; then
    pass "E2E: switch reason is valid ($STATE_REASON)"
  else
    fail "E2E: unexpected switch reason ($STATE_REASON)"
  fi
else
  fail "E2E: curriculum_state.json not created"
fi

# Startup grace messages
if grep -q "Waiting for first metric" "$TEST_DIR/supervisor_output.log" 2>/dev/null; then
  pass "E2E: startup grace period log messages present"
else
  fail "E2E: no startup grace period messages in log"
fi

if grep -q "First metric received" "$TEST_DIR/supervisor_output.log" 2>/dev/null; then
  pass "E2E: 'First metric received' transition logged"
else
  fail "E2E: 'First metric received' not found in log"
fi

# State file fields
if [[ -f "$MOCK_EXP/curriculum_state.json" ]]; then
  FIELDS_OK=$(python3 -c "
import json
with open('$MOCK_EXP/curriculum_state.json') as f:
  s = json.load(f)
required = ['current_book', 'book_index', 'switch_reason', 'switch_time', 'loss_at_switch', 'total_books']
missing = [f for f in required if f not in s]
if missing:
  print(f'MISSING: {missing}')
else:
  print('OK')
")
  if [[ "$FIELDS_OK" == "OK" ]]; then
    pass "E2E: curriculum_state.json has all required fields"
  else
    fail "E2E: curriculum_state.json $FIELDS_OK"
  fi
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 10: Metrics.jsonl symlink created ==="
# ──────────────────────────────────────────────────────────────────────────────

if [[ -L "$MOCK_EXP/metrics.jsonl" ]]; then
  LINK_TARGET=$(readlink "$MOCK_EXP/metrics.jsonl")
  if [[ "$LINK_TARGET" == "checkpoints/metrics.jsonl" ]]; then
    pass "Symlink: metrics.jsonl -> checkpoints/metrics.jsonl"
  else
    fail "Symlink target wrong: $LINK_TARGET (expected checkpoints/metrics.jsonl)"
  fi
else
  fail "metrics.jsonl symlink not created at experiment root"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 11: Startup timeout (no metrics produced) ==="
# ──────────────────────────────────────────────────────────────────────────────

MOCK_BIN_HANG="$TEST_DIR/mock-erm-hang"
cat > "$MOCK_BIN_HANG" << 'MOCKEOF'
#!/usr/bin/env bash
CKPT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir) CKPT_DIR="$2"; shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$CKPT_DIR"
handle_term() { exit 0; }
trap handle_term SIGTERM
while true; do sleep 0.5; done
MOCKEOF
chmod +x "$MOCK_BIN_HANG"

MOCK_EXP_HANG="$TEST_DIR/experiment_hang"
mkdir -p "$MOCK_EXP_HANG"

echo "[test11] Running supervisor with 6s startup timeout, mock binary that never writes metrics..."

(
  "$SCRIPT_DIR/curriculum-supervisor.sh" \
    --books-dir "$MOCK_BOOKS" \
    --experiment-dir "$MOCK_EXP_HANG" \
    --config "$MOCK_CONFIG" \
    --erm-bin "$MOCK_BIN_HANG" \
    --backend cpu \
    --steps-per-book 10000 \
    --log-every 50 \
    --checkpoint-every 250 \
    --startup-timeout 6 \
    --plateau-window 3 \
    --plateau-threshold 0.005 \
    --hard-timeout 30 \
    --poll-interval 2 \
    --max-books 1 \
    --exp-id-prefix "hang" \
    --bpe-num-merges 50 \
    --bpe-sample-mb 1
) > "$TEST_DIR/hang_output.log" 2>&1 &
HANG_PID=$!

WAITED=0
while kill -0 "$HANG_PID" 2>/dev/null && [[ $WAITED -lt 60 ]]; do
  sleep 2
  WAITED=$((WAITED + 2))
done
if kill -0 "$HANG_PID" 2>/dev/null; then
  kill -9 "$HANG_PID" 2>/dev/null || true
fi
wait "$HANG_PID" 2>/dev/null || true

if grep -q "STARTUP TIMEOUT" "$TEST_DIR/hang_output.log" 2>/dev/null; then
  pass "Startup timeout: correctly triggered after no metrics"
else
  fail "Startup timeout: not triggered"
fi

if [[ -f "$MOCK_EXP_HANG/curriculum_state.json" ]]; then
  HANG_REASON=$(python3 -c "
import json
with open('$MOCK_EXP_HANG/curriculum_state.json') as f:
  s = json.load(f)
print(s.get('switch_reason', 'unknown'))
")
  if [[ "$HANG_REASON" == "startup_timeout" ]]; then
    pass "Startup timeout: state reason = startup_timeout"
  else
    fail "Startup timeout: state reason = $HANG_REASON (expected startup_timeout)"
  fi
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 12: No misleading 'still improving' during startup ==="
# ──────────────────────────────────────────────────────────────────────────────

if grep -q "loss=0.000000, still improving" "$TEST_DIR/supervisor_output.log" 2>/dev/null; then
  fail "Misleading 'loss=0.000000, still improving' found during startup"
else
  pass "No misleading 'still improving' messages during startup"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 13: Universal BPE vocabulary built ==="
# ──────────────────────────────────────────────────────────────────────────────

VOCAB_FILE="$MOCK_EXP/curriculum_vocab/bpe_vocab.json"
if [[ -f "$VOCAB_FILE" ]]; then
  VOCAB_CHECK=$(python3 -c "
import json
with open('$VOCAB_FILE') as f:
    v = json.load(f)
ok = True
if 'merges' not in v: ok = False
if 'vocab' not in v: ok = False
if 'id_to_token' not in v: ok = False
if 'vocab_size' not in v: ok = False
if v.get('vocab_size', 0) < 10: ok = False
print('OK' if ok else 'BAD')
print(f'vocab_size={v.get(\"vocab_size\", 0)}')
")
  if echo "$VOCAB_CHECK" | head -1 | grep -q "OK"; then
    VSIZE=$(echo "$VOCAB_CHECK" | tail -1)
    pass "Universal BPE vocab built ($VSIZE)"
  else
    fail "Universal BPE vocab file malformed: $VOCAB_CHECK"
  fi
else
  fail "Universal BPE vocab not created at $VOCAB_FILE"
fi

# Check patched config references the vocab
PATCHED_CONFIG="$MOCK_EXP/patched_config.json"
if [[ -f "$PATCHED_CONFIG" ]]; then
  BPE_PATH=$(python3 -c "
import json
with open('$PATCHED_CONFIG') as f:
    cfg = json.load(f)
print(cfg.get('bpe_vocab_path', ''))
")
  if [[ -n "$BPE_PATH" ]] && [[ "$BPE_PATH" == *"bpe_vocab.json"* ]]; then
    pass "Patched config has bpe_vocab_path set"
  else
    fail "Patched config bpe_vocab_path not set (got: $BPE_PATH)"
  fi
else
  fail "Patched config not created"
fi

# Verify vocab was logged
if grep -q "Universal BPE vocabulary" "$TEST_DIR/supervisor_output.log" 2>/dev/null; then
  pass "Vocab building logged in supervisor output"
else
  fail "No vocab building messages in supervisor log"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 14: Pre-existing vocab is reused (not rebuilt) ==="
# ──────────────────────────────────────────────────────────────────────────────

# Re-run with the same experiment dir — vocab should already exist
MOCK_EXP_REUSE="$TEST_DIR/experiment_reuse"
mkdir -p "$MOCK_EXP_REUSE/curriculum_vocab"

# Copy the vocab from test 13
cp "$VOCAB_FILE" "$MOCK_EXP_REUSE/curriculum_vocab/bpe_vocab.json"

(
  "$SCRIPT_DIR/curriculum-supervisor.sh" \
    --books-dir "$MOCK_BOOKS" \
    --experiment-dir "$MOCK_EXP_REUSE" \
    --config "$MOCK_CONFIG" \
    --erm-bin "$MOCK_BIN" \
    --backend cpu \
    --steps-per-book 10000 \
    --log-every 50 \
    --checkpoint-every 250 \
    --startup-timeout 30 \
    --plateau-window 3 \
    --plateau-threshold 0.005 \
    --hard-timeout 15 \
    --poll-interval 2 \
    --max-books 1 \
    --exp-id-prefix "reuse" \
    --bpe-num-merges 50 \
    --bpe-sample-mb 1
) > "$TEST_DIR/reuse_output.log" 2>&1 &
REUSE_PID=$!

WAITED=0
while kill -0 "$REUSE_PID" 2>/dev/null && [[ $WAITED -lt 60 ]]; do
  sleep 2
  WAITED=$((WAITED + 2))
done
if kill -0 "$REUSE_PID" 2>/dev/null; then
  kill -9 "$REUSE_PID" 2>/dev/null || true
fi
wait "$REUSE_PID" 2>/dev/null || true

if grep -q "Universal BPE vocabulary found" "$TEST_DIR/reuse_output.log" 2>/dev/null; then
  pass "Pre-existing vocab was detected and reused"
else
  fail "Pre-existing vocab was not reused"
fi

if grep -q "Building from full corpus" "$TEST_DIR/reuse_output.log" 2>/dev/null; then
  fail "Vocab was rebuilt despite already existing"
else
  pass "Vocab was NOT rebuilt (correctly skipped)"
fi

# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed"
echo "═══════════════════════════════════════════"

if [[ $FAIL -gt 0 ]]; then
  echo ""
  echo "=== Supervisor output (Test 9, last 40 lines) ==="
  tail -40 "$TEST_DIR/supervisor_output.log" 2>/dev/null || echo "(no output)"
  echo ""
  echo "=== Hang test output (Test 11, last 20 lines) ==="
  tail -20 "$TEST_DIR/hang_output.log" 2>/dev/null || echo "(no output)"
  echo ""
  echo "=== Reuse test output (Test 14, last 20 lines) ==="
  tail -20 "$TEST_DIR/reuse_output.log" 2>/dev/null || echo "(no output)"
  exit 1
fi
exit 0
