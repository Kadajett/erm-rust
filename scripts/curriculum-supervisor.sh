#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# curriculum-supervisor.sh — Auto-curriculum book switcher for ERM diffusion training
#
# Watches metrics.jsonl for loss plateau and cycles through books in a corpus
# directory. Has a startup grace period to allow CUDA JIT warmup to complete
# before monitoring begins. On plateau (loss delta < 0.5% over 5min window) OR
# after a hard training timeout, sends SIGTERM to the training process
# (triggering checkpoint save), then relaunches with the next book.
#
# Universal BPE vocabulary:
#   Before training starts, checks for a pre-built BPE vocabulary. If missing,
#   runs build-universal-vocab.py to train BPE on the full corpus once. This
#   vocabulary is then passed to every training run via --config (bpe_vocab_path),
#   ensuring consistent token IDs across books.
#
# Timeout model:
#   1. Startup grace period (--startup-timeout, default 600s/10min)
#      — waits for first non-zero metric line in metrics.jsonl
#      — if nothing after startup-timeout, skip the book (broken)
#   2. Training plateau timeout (--plateau-window, default 300s/5min)
#      — only starts AFTER first real metric appears
#      — flat loss for this window triggers a book switch
#   3. Hard training timeout (--hard-timeout, default 900s/15min)
#      — wall-clock from first metric, NOT including startup
#      — backstop if loss keeps dropping but book is taking too long
#
# Usage:
#   curriculum-supervisor.sh \
#     --books-dir /workspace/rust-pcn/data/books \
#     --experiment-dir /workspace/erm-rust/data/experiments/alice-run \
#     --config /workspace/erm-rust/data/alice_4gib_config.json \
#     --erm-bin /workspace/erm-rust/bin/erm \
#     --backend cuda \
#     --steps-per-book 10000 \
#     --log-every 50 \
#     --checkpoint-every 250
#
# State file: curriculum_state.json (written alongside --experiment-dir)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
BOOKS_DIR=""
EXPERIMENT_DIR=""
CONFIG_PATH=""
ERM_BIN="/workspace/erm-rust/bin/erm"
BACKEND="cuda"
STEPS_PER_BOOK=10000
LOG_EVERY=50
CHECKPOINT_EVERY=250
STARTUP_TIMEOUT_SECS=600      # 10 minutes to get first metric
PLATEAU_WINDOW_SECS=300        # 5 minutes of flat loss after training starts
PLATEAU_THRESHOLD=0.005        # 0.5% improvement required
HARD_TIMEOUT_SECS=900          # 15-minute hard timeout AFTER training starts
POLL_INTERVAL_SECS=15          # how often to check metrics
MAX_BOOKS=0                    # 0 = all books in directory
EXP_ID_PREFIX="curriculum"
BPE_VOCAB_PATH=""              # auto-detected if empty
BPE_NUM_MERGES=4096
BPE_SAMPLE_MB=10

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --books-dir)         BOOKS_DIR="$2";             shift 2 ;;
    --experiment-dir)    EXPERIMENT_DIR="$2";         shift 2 ;;
    --config)            CONFIG_PATH="$2";            shift 2 ;;
    --erm-bin)           ERM_BIN="$2";                shift 2 ;;
    --backend)           BACKEND="$2";                shift 2 ;;
    --steps-per-book)    STEPS_PER_BOOK="$2";         shift 2 ;;
    --log-every)         LOG_EVERY="$2";              shift 2 ;;
    --checkpoint-every)  CHECKPOINT_EVERY="$2";       shift 2 ;;
    --startup-timeout)   STARTUP_TIMEOUT_SECS="$2";   shift 2 ;;
    --plateau-window)    PLATEAU_WINDOW_SECS="$2";    shift 2 ;;
    --plateau-threshold) PLATEAU_THRESHOLD="$2";      shift 2 ;;
    --hard-timeout)      HARD_TIMEOUT_SECS="$2";      shift 2 ;;
    --poll-interval)     POLL_INTERVAL_SECS="$2";     shift 2 ;;
    --max-books)         MAX_BOOKS="$2";              shift 2 ;;
    --exp-id-prefix)     EXP_ID_PREFIX="$2";          shift 2 ;;
    --bpe-vocab-path)    BPE_VOCAB_PATH="$2";         shift 2 ;;
    --bpe-num-merges)    BPE_NUM_MERGES="$2";         shift 2 ;;
    --bpe-sample-mb)     BPE_SAMPLE_MB="$2";          shift 2 ;;
    *)                   echo "Unknown arg: $1" >&2;  exit 1 ;;
  esac
done

# ── Validate ─────────────────────────────────────────────────────────────────
[[ -z "$BOOKS_DIR" ]]       && { echo "ERROR: --books-dir required" >&2; exit 1; }
[[ -z "$EXPERIMENT_DIR" ]]  && { echo "ERROR: --experiment-dir required" >&2; exit 1; }
[[ -z "$CONFIG_PATH" ]]     && { echo "ERROR: --config required" >&2; exit 1; }
[[ ! -d "$BOOKS_DIR" ]]     && { echo "ERROR: books dir not found: $BOOKS_DIR" >&2; exit 1; }
[[ ! -x "$ERM_BIN" ]]       && { echo "ERROR: erm binary not found/executable: $ERM_BIN" >&2; exit 1; }

# ── Build sorted book list ───────────────────────────────────────────────────
mapfile -t ALL_BOOKS < <(find "$BOOKS_DIR" -maxdepth 1 -type f -name '*.txt' | sort)
TOTAL_BOOKS=${#ALL_BOOKS[@]}

if [[ $TOTAL_BOOKS -eq 0 ]]; then
  echo "ERROR: no .txt files found in $BOOKS_DIR" >&2
  exit 1
fi

if [[ $MAX_BOOKS -gt 0 && $MAX_BOOKS -lt $TOTAL_BOOKS ]]; then
  TOTAL_BOOKS=$MAX_BOOKS
fi

echo "=== Curriculum Supervisor ==="
echo "Books dir: $BOOKS_DIR ($TOTAL_BOOKS books)"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Config: $CONFIG_PATH"
echo "Backend: $BACKEND"
echo "Steps per book: $STEPS_PER_BOOK"
echo "Startup grace timeout: ${STARTUP_TIMEOUT_SECS}s"
echo "Plateau window: ${PLATEAU_WINDOW_SECS}s, threshold: ${PLATEAU_THRESHOLD}"
echo "Hard training timeout: ${HARD_TIMEOUT_SECS}s (after first metric)"
echo "Poll interval: ${POLL_INTERVAL_SECS}s"

mkdir -p "$EXPERIMENT_DIR"

# ── Universal BPE Vocabulary ─────────────────────────────────────────────────
# Ensure a universal BPE vocabulary exists before training any books.
# This prevents token ID shifts between books.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VOCAB_DIR="$EXPERIMENT_DIR/curriculum_vocab"

# Determine vocab path
if [[ -n "$BPE_VOCAB_PATH" ]]; then
  UNIVERSAL_VOCAB="$BPE_VOCAB_PATH"
else
  UNIVERSAL_VOCAB="$VOCAB_DIR/bpe_vocab.json"
fi

if [[ -f "$UNIVERSAL_VOCAB" ]]; then
  VOCAB_SIZE=$(python3 -c "
import json
with open('$UNIVERSAL_VOCAB') as f:
  v = json.load(f)
print(v.get('vocab_size', 'unknown'))
" 2>/dev/null || echo "unknown")
  echo "[vocab] Universal BPE vocabulary found: $UNIVERSAL_VOCAB (vocab_size=$VOCAB_SIZE)"
else
  echo "[vocab] No universal BPE vocabulary found at: $UNIVERSAL_VOCAB"
  echo "[vocab] Building from full corpus ($TOTAL_BOOKS books, ${BPE_NUM_MERGES} merges, ${BPE_SAMPLE_MB}MB sample)..."
  mkdir -p "$VOCAB_DIR"

  BUILD_SCRIPT="$SCRIPT_DIR/build-universal-vocab.py"
  if [[ ! -f "$BUILD_SCRIPT" ]]; then
    echo "ERROR: build-universal-vocab.py not found at $BUILD_SCRIPT" >&2
    exit 1
  fi

  BPE_START=$(date +%s)
  python3 "$BUILD_SCRIPT" \
    --books-dir "$BOOKS_DIR" \
    --output "$UNIVERSAL_VOCAB" \
    --num-merges "$BPE_NUM_MERGES" \
    --sample-mb "$BPE_SAMPLE_MB"

  BPE_END=$(date +%s)
  BPE_ELAPSED=$((BPE_END - BPE_START))
  echo "[vocab] Universal BPE vocabulary built in ${BPE_ELAPSED}s: $UNIVERSAL_VOCAB"

  VOCAB_SIZE=$(python3 -c "
import json
with open('$UNIVERSAL_VOCAB') as f:
  v = json.load(f)
print(v.get('vocab_size', 'unknown'))
" 2>/dev/null || echo "unknown")
  echo "[vocab] Vocab size: $VOCAB_SIZE"
fi

# ── Patch config to use universal vocabulary ─────────────────────────────────
# Create a patched config that points bpe_vocab_path to the universal vocab.
# This ensures the erm binary loads the pre-built vocab instead of training one.
PATCHED_CONFIG="$EXPERIMENT_DIR/patched_config.json"
python3 -c "
import json, sys
with open('$CONFIG_PATH') as f:
    cfg = json.load(f)
cfg['bpe_vocab_path'] = '$UNIVERSAL_VOCAB'
with open('$PATCHED_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'[vocab] Patched config: bpe_vocab_path = {cfg[\"bpe_vocab_path\"]}')
" || { echo "ERROR: failed to patch config" >&2; exit 1; }

# Use patched config for all training runs
CONFIG_PATH="$PATCHED_CONFIG"

# ── State file ───────────────────────────────────────────────────────────────
STATE_FILE="$EXPERIMENT_DIR/curriculum_state.json"

# Load or initialize state
BOOK_INDEX=0
if [[ -f "$STATE_FILE" ]]; then
  SAVED_INDEX=$(python3 -c "
import json, sys
try:
  with open('$STATE_FILE') as f:
    s = json.load(f)
  print(s.get('book_index', 0))
except:
  print(0)
" 2>/dev/null || echo "0")
  BOOK_INDEX=$SAVED_INDEX
  echo "Resuming from book index: $BOOK_INDEX"
fi

# ── Helper: write state ─────────────────────────────────────────────────────
write_state() {
  local book_path="$1"
  local book_idx="$2"
  local switch_reason="$3"
  local loss_at_switch="$4"

  cat > "$STATE_FILE" <<EOF
{
  "current_book": "$(basename "$book_path")",
  "current_book_path": "$book_path",
  "book_index": $book_idx,
  "total_books": $TOTAL_BOOKS,
  "switch_reason": "$switch_reason",
  "switch_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "loss_at_switch": $loss_at_switch,
  "books_dir": "$BOOKS_DIR",
  "experiment_dir": "$EXPERIMENT_DIR",
  "bpe_vocab_path": "$UNIVERSAL_VOCAB",
  "startup_timeout_secs": $STARTUP_TIMEOUT_SECS,
  "plateau_window_secs": $PLATEAU_WINDOW_SECS,
  "plateau_threshold": $PLATEAU_THRESHOLD,
  "hard_timeout_secs": $HARD_TIMEOUT_SECS
}
EOF
  echo "[state] Wrote curriculum_state.json: book=$book_idx reason=$switch_reason loss=$loss_at_switch"
}

# ── Helper: check if metrics file has a non-zero loss line ───────────────────
# Returns 0 (true) if a real metric exists, 1 (false) if not.
# Prints the latest loss to stdout.
has_real_metric() {
  local metrics_file="$1"

  if [[ ! -f "$metrics_file" ]] || [[ ! -s "$metrics_file" ]]; then
    echo "0.0"
    return 1
  fi

  python3 -c "
import json, sys

metrics_file = '$metrics_file'
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

if not records:
    print('0.0')
    sys.exit(1)

# Check if any record has loss != 0.0
for r in records:
    loss = r.get('loss', 0.0)
    if loss != 0.0 and loss != 0:
        # Has real metric — print latest loss
        print(f'{records[-1][\"loss\"]:.6f}')
        sys.exit(0)

# All losses are 0.0 — no real metric yet
print('0.0')
sys.exit(1)
" 2>/dev/null
}

# ── Helper: detect plateau from metrics.jsonl ────────────────────────────────
# Returns 0 (true) if plateaued, 1 (false) if still improving.
# Also prints the current loss to stdout.
detect_plateau() {
  local metrics_file="$1"
  local window_secs="$2"
  local threshold="$3"

  if [[ ! -f "$metrics_file" ]]; then
    echo "0.0"
    return 1  # no metrics yet → not plateaued
  fi

  python3 -c "
import json, sys, time

metrics_file = '$metrics_file'
window_secs = $window_secs
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
    # Not enough data
    loss = records[-1]['loss'] if records else 0.0
    print(f'{loss:.6f}')
    sys.exit(1)

# Look at the last N records for plateau detection
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

# ── Helper: get current loss from metrics ────────────────────────────────────
get_current_loss() {
  local metrics_file="$1"
  if [[ -f "$metrics_file" ]]; then
    tail -1 "$metrics_file" 2>/dev/null | python3 -c "
import json, sys
try:
  r = json.loads(sys.stdin.read().strip())
  print(f\"{r['loss']:.6f}\")
except:
  print('0.0')
" 2>/dev/null || echo "0.0"
  else
    echo "0.0"
  fi
}

# ── Helper: gracefully stop a process ────────────────────────────────────────
stop_process() {
  local pid="$1"
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  echo "[supervisor] Sending SIGTERM to PID=$pid"
  kill -TERM "$pid" 2>/dev/null || true

  local waited=0
  while [ $waited -lt 60 ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[supervisor] Process exited after SIGTERM (${waited}s)"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done

  echo "[supervisor] Force killing PID=$pid"
  kill -9 "$pid" 2>/dev/null || true
  sleep 1
  return 0
}

# ── Helper: check if process is alive ────────────────────────────────────────
is_alive() {
  kill -0 "$1" 2>/dev/null
}

# ── Helper: run io-example on new checkpoints asynchronously ─────────────────
# Scans for step_NNNNN/ checkpoint dirs that lack io_example.json, and runs
# erm io-example in the background (CPU only, won't block GPU training).
IO_EXAMPLE_PIDS=()

run_io_examples_async() {
  local ckpt_base="$1"
  local data_file="$2"

  # Clean up completed io-example PIDs
  local still_running=()
  for pid in "${IO_EXAMPLE_PIDS[@]}"; do
    if is_alive "$pid"; then
      still_running+=("$pid")
    fi
  done
  IO_EXAMPLE_PIDS=("${still_running[@]}")

  # Don't launch more than 1 concurrent io-example (CPU bound, slow)
  if [[ ${#IO_EXAMPLE_PIDS[@]} -ge 1 ]]; then
    return
  fi

  # Scan for step dirs missing io_example.json
  for step_dir in "$ckpt_base"/step_*/; do
    [[ -d "$step_dir" ]] || continue
    [[ -f "$step_dir/io_example.json" ]] && continue
    # Must have a scorer.bin to be a valid checkpoint
    [[ -f "$step_dir/scorer.bin" ]] || continue

    local step_name
    step_name=$(basename "$step_dir")
    echo "[io-example] Found checkpoint without IO examples: $step_name"

    (
      "$ERM_BIN" io-example \
        --data "$data_file" \
        --checkpoint "$step_dir" \
        --output "$step_dir/io_example.json" \
        --num-examples 3 \
        --seed 42 \
        2>&1 | while IFS= read -r line; do echo "[io-example:$step_name] $line"; done

      # Copy to latest/ for the watcher to pick up
      if [[ -f "$step_dir/io_example.json" ]]; then
        cp "$step_dir/io_example.json" "$ckpt_base/latest/io_example.json" 2>/dev/null || true
        echo "[io-example] Wrote $step_dir/io_example.json and copied to latest/"
      fi
    ) &

    IO_EXAMPLE_PIDS+=($!)
    echo "[io-example] Launched background io-example PID=$! for $step_name"
    # Only launch one at a time
    break
  done
}

# ── Helper: cleanup io-example background jobs ──────────────────────────────
cleanup_io_examples() {
  for pid in "${IO_EXAMPLE_PIDS[@]}"; do
    if is_alive "$pid"; then
      echo "[io-example] Killing background io-example PID=$pid"
      kill "$pid" 2>/dev/null || true
    fi
  done
  IO_EXAMPLE_PIDS=()
}

# ── Main training loop ──────────────────────────────────────────────────────
TRAINING_PID=0

cleanup() {
  if [[ $TRAINING_PID -gt 0 ]] && is_alive "$TRAINING_PID"; then
    echo "[supervisor] Caught signal, stopping training PID=$TRAINING_PID"
    kill -TERM "$TRAINING_PID" 2>/dev/null || true
    wait "$TRAINING_PID" 2>/dev/null || true
  fi
  cleanup_io_examples
  exit 0
}
trap cleanup SIGTERM SIGINT

while [[ $BOOK_INDEX -lt $TOTAL_BOOKS ]]; do
  BOOK_PATH="${ALL_BOOKS[$BOOK_INDEX]}"
  BOOK_NAME=$(basename "$BOOK_PATH")
  EXP_ID="${EXP_ID_PREFIX}-book${BOOK_INDEX}"
  CKPT_DIR="$EXPERIMENT_DIR/checkpoints"
  METRICS_FILE="$CKPT_DIR/metrics.jsonl"

  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  Book $((BOOK_INDEX + 1))/$TOTAL_BOOKS: $BOOK_NAME"
  echo "║  Index: $BOOK_INDEX | Exp: $EXP_ID"
  echo "║  Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "║  Vocab: $UNIVERSAL_VOCAB (shared across all books)"
  echo "╚══════════════════════════════════════════════════════════════╝"

  # Copy the book to a staging dir
  BOOK_STAGE_DIR="$EXPERIMENT_DIR/current_book"
  mkdir -p "$BOOK_STAGE_DIR"
  cp "$BOOK_PATH" "$BOOK_STAGE_DIR/current.txt"
  echo "Book size: $(stat -c '%s' "$BOOK_STAGE_DIR/current.txt" 2>/dev/null || stat -f '%z' "$BOOK_STAGE_DIR/current.txt" 2>/dev/null) bytes"

  # Clear old metrics for this book session (we track per-book plateau)
  if [[ -f "$METRICS_FILE" ]]; then
    BACKUP="$EXPERIMENT_DIR/metrics_book$((BOOK_INDEX - 1))_$(date +%s).jsonl"
    cp "$METRICS_FILE" "$BACKUP" 2>/dev/null || true
    echo "[supervisor] Backed up previous metrics to $BACKUP"
    > "$METRICS_FILE"
  fi

  mkdir -p "$CKPT_DIR"

  # Create/maintain metrics.jsonl symlink at experiment root for dashboard
  METRICS_SYMLINK="$EXPERIMENT_DIR/metrics.jsonl"
  if [[ -L "$METRICS_SYMLINK" ]]; then
    rm -f "$METRICS_SYMLINK"
  elif [[ -e "$METRICS_SYMLINK" ]]; then
    mv "$METRICS_SYMLINK" "$METRICS_SYMLINK.bak" 2>/dev/null || true
  fi
  ln -s "checkpoints/metrics.jsonl" "$METRICS_SYMLINK"
  echo "[supervisor] metrics.jsonl symlink: $METRICS_SYMLINK -> checkpoints/metrics.jsonl"

  # Write initial state
  write_state "$BOOK_PATH" "$BOOK_INDEX" "started" "0.0"

  # Launch training in background
  echo "[supervisor] Launching: $ERM_BIN diffusion-train --data $BOOK_STAGE_DIR/current.txt --config $CONFIG_PATH"

  "$ERM_BIN" diffusion-train \
    --data "$BOOK_STAGE_DIR/current.txt" \
    --steps "$STEPS_PER_BOOK" \
    --config "$CONFIG_PATH" \
    --backend "$BACKEND" \
    --log-every "$LOG_EVERY" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --checkpoint-dir "$CKPT_DIR" \
    --exp-id "$EXP_ID" \
    > "$EXPERIMENT_DIR/console_book${BOOK_INDEX}.log" 2>&1 &

  TRAINING_PID=$!
  BOOK_START_TIME=$(date +%s)
  TRAINING_STARTED=0
  TRAINING_START_TIME=0
  SWITCH_REASON=""

  echo "[supervisor] Training PID=$TRAINING_PID, waiting for first metric (startup grace: ${STARTUP_TIMEOUT_SECS}s)..."

  # ── Monitor loop ────────────────────────────────────────────────────
  while true; do
    sleep "$POLL_INTERVAL_SECS"

    # Check if training already exited
    if ! is_alive "$TRAINING_PID"; then
      echo "[supervisor] Training process exited"
      CURRENT_LOSS=$(get_current_loss "$METRICS_FILE")
      SWITCH_REASON="training_completed"
      break
    fi

    NOW=$(date +%s)
    ELAPSED_FROM_LAUNCH=$((NOW - BOOK_START_TIME))

    # ── Phase 1: Startup grace period (waiting for first metric) ──
    if [[ $TRAINING_STARTED -eq 0 ]]; then
      set +e
      FIRST_LOSS=$(has_real_metric "$METRICS_FILE")
      HAS_METRIC=$?
      set -e

      if [[ $HAS_METRIC -eq 0 ]]; then
        TRAINING_STARTED=1
        TRAINING_START_TIME=$NOW
        echo "[supervisor] First metric received at ${ELAPSED_FROM_LAUNCH}s after launch (loss=$FIRST_LOSS)"
        echo "[supervisor] Training phase started — plateau window: ${PLATEAU_WINDOW_SECS}s, hard timeout: ${HARD_TIMEOUT_SECS}s"
        continue
      fi

      # Still waiting
      if [[ $ELAPSED_FROM_LAUNCH -ge $STARTUP_TIMEOUT_SECS ]]; then
        echo "[supervisor] STARTUP TIMEOUT: no metrics after ${ELAPSED_FROM_LAUNCH}s (limit: ${STARTUP_TIMEOUT_SECS}s) — skipping book"
        CURRENT_LOSS="0.0"
        SWITCH_REASON="startup_timeout"
        stop_process "$TRAINING_PID"
        wait "$TRAINING_PID" 2>/dev/null || true
        break
      fi

      echo "[supervisor] Waiting for first metric (${ELAPSED_FROM_LAUNCH}s / ${STARTUP_TIMEOUT_SECS}s elapsed)..."
      continue
    fi

    # ── Phase 2: Training active — monitor plateau and hard timeout ──
    TRAINING_ELAPSED=$((NOW - TRAINING_START_TIME))
    
    # Run IO examples for any new checkpoints
    run_io_examples_async "$CKPT_DIR" "$BOOK_STAGE_DIR/current.txt"

    # Check hard training timeout
    if [[ $TRAINING_ELAPSED -ge $HARD_TIMEOUT_SECS ]]; then
      echo "[supervisor] Hard training timeout reached (${TRAINING_ELAPSED}s >= ${HARD_TIMEOUT_SECS}s of actual training)"
      CURRENT_LOSS=$(get_current_loss "$METRICS_FILE")
      SWITCH_REASON="hard_timeout"
      stop_process "$TRAINING_PID"
      wait "$TRAINING_PID" 2>/dev/null || true
      break
    fi

    # Check plateau (only after at least 60s of actual training)
    if [[ $TRAINING_ELAPSED -ge 60 ]]; then
      set +e
      CURRENT_LOSS=$(detect_plateau "$METRICS_FILE" "$PLATEAU_WINDOW_SECS" "$PLATEAU_THRESHOLD")
      PLATEAU_EXIT=$?
      set -e

      if [[ $PLATEAU_EXIT -eq 0 ]]; then
        echo "[supervisor] Plateau detected at ${TRAINING_ELAPSED}s of training (loss=$CURRENT_LOSS)"
        SWITCH_REASON="plateau_detected"
        stop_process "$TRAINING_PID"
        wait "$TRAINING_PID" 2>/dev/null || true
        break
      else
        echo "[supervisor] Book $BOOK_INDEX: ${TRAINING_ELAPSED}s training, loss=$CURRENT_LOSS, still improving"
      fi
    else
      CURRENT_LOSS=$(get_current_loss "$METRICS_FILE")
      echo "[supervisor] Book $BOOK_INDEX: ${TRAINING_ELAPSED}s training (warming up), loss=$CURRENT_LOSS"
    fi
  done

  # Ensure we reap the child
  wait "$TRAINING_PID" 2>/dev/null || true
  TRAINING_PID=0

  # Write final state for this book
  CURRENT_LOSS=$(get_current_loss "$METRICS_FILE")
  write_state "$BOOK_PATH" "$BOOK_INDEX" "$SWITCH_REASON" "$CURRENT_LOSS"

  echo "[supervisor] Book $BOOK_INDEX done: reason=$SWITCH_REASON loss=$CURRENT_LOSS"

  # Advance to next book
  BOOK_INDEX=$((BOOK_INDEX + 1))

  # Small pause between books to let GPU memory settle
  sleep 2
done

echo ""
echo "=== Curriculum complete ==="
echo "Processed $BOOK_INDEX / $TOTAL_BOOKS books"
echo "Universal vocab: $UNIVERSAL_VOCAB"
echo "Final state: $STATE_FILE"
cat "$STATE_FILE"
