#!/usr/bin/env python3
"""
Aim Watcher — tails metrics.jsonl and streams to Aim server.

Single Aim Run for the entire curriculum training (all 203 books).
Book switches are detected via curriculum_state.json and logged as
run hyperparameters (book_{N} = name, loss_at_book_{N}_switch = loss).

Features:
  - Metrics tracked with context dicts (subset, book) for proper Aim grouping
  - Run params populated from config.json and curriculum_state.json
  - Per-book loss statistics: distribution, delta, start/end/min
  - Ant colony state tracking from ant_state.json
  - Loss-per-edit and edit-rate as accuracy proxies
  - Plateau detection with Text event logging

Modes:
  watch   (default) — tail metrics.jsonl, push live metrics
  backfill          — read all metrics_book*.jsonl backup files, push historical data

Env vars:
  AIM_SERVER_HOST  — hostname/IP of the Aim server (e.g. aim-server)
  AIM_SERVER_PORT  — port (default 43800)
  EXPERIMENT_DIR   — path to the curriculum-run experiment directory
"""

import glob
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("aim-watcher")

from aim import Run, Distribution, Text

AIM_SERVER = os.environ.get("AIM_SERVER_HOST", "aim-server")
AIM_PORT = int(os.environ.get("AIM_SERVER_PORT", "43800"))
EXPERIMENT_DIR = os.environ.get(
    "EXPERIMENT_DIR",
    "/data/experiments/curriculum-run",
)
METRICS_FILE = os.path.join(EXPERIMENT_DIR, "checkpoints", "metrics.jsonl")
CURRICULUM_STATE_FILE = os.path.join(EXPERIMENT_DIR, "curriculum_state.json")
CONFIG_FILE = os.path.join(EXPERIMENT_DIR, "checkpoints", "latest", "config.json")
ANT_STATE_FILE = os.path.join(EXPERIMENT_DIR, "checkpoints", "latest", "ant_state.json")
STEP_FILE = os.path.join(EXPERIMENT_DIR, "checkpoints", "latest", "step.json")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "5"))

AIM_REPO = f"aim://{AIM_SERVER}:{AIM_PORT}"
AIM_EXPERIMENT = os.environ.get("AIM_EXPERIMENT", "erm-training")

TRACKED_METRICS = ["loss", "edits", "mean_phi", "deaths"]

# Plateau detection: if loss variance is below this over N readings, flag it
PLATEAU_WINDOW = 10
PLATEAU_THRESHOLD = 0.005


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        log.warning("Bad JSON line: %s — %s", line[:120], e)
        return None


def load_json_file(filepath: str) -> dict | None:
    """Safely load a JSON file, return None on any error."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        log.debug("Could not load %s: %s", filepath, e)
        return None


def load_curriculum_state() -> dict | None:
    return load_json_file(CURRICULUM_STATE_FILE)


def load_config() -> dict | None:
    return load_json_file(CONFIG_FILE)


def load_ant_state() -> dict | None:
    return load_json_file(ANT_STATE_FILE)


def load_step() -> int:
    data = load_json_file(STEP_FILE)
    if data and "step" in data:
        return data["step"]
    return 0


def infer_experiment_id() -> str:
    """Infer stable experiment identifier from config, metrics, or directory name."""
    config = load_config()
    if config and config.get("exp_id"):
        return str(config["exp_id"])

    # Fall back to first metrics line if config isn't present yet.
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                for line in f:
                    record = parse_line(line)
                    if record and record.get("exp_id"):
                        return str(record["exp_id"])
                    # Only inspect the first non-empty parseable line.
                    if record:
                        break
        except OSError as e:
            log.debug("Could not infer exp_id from metrics: %s", e)

    return Path(EXPERIMENT_DIR).name


def build_run_hash(experiment_id: str) -> str:
    """Build deterministic Aim run hash (24 hex chars) for this experiment."""
    key = f"{AIM_EXPERIMENT}|{EXPERIMENT_DIR}|{experiment_id}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]


def try_add_tag(run: Run, tag: str):
    """Best-effort Aim tag writer."""
    if not tag:
        return
    try:
        run.add_tag(tag)
    except Exception as e:
        log.debug("Could not add Aim tag '%s': %s", tag, e)


def set_run_params(run: Run):
    """Populate run params from config.json and curriculum_state.json."""
    config = load_config()
    if config:
        run["config"] = {
            "hidden_dim": config.get("hidden_dim", 256),
            "seq_len": config.get("seq_len", 256),
            "batch_size": config.get("batch_size", 1),
            "num_blocks": config.get("num_blocks", 4),
            "num_heads": config.get("num_heads", 8),
            "num_ants": config.get("num_ants", 128),
            "vocab_size": config.get("vocab_size", 50257),
            "tokenizer_type": config.get("tokenizer_type", "bpe"),
            "learning_rate": config.get("learning_rate", 0.0003),
            "weight_decay": config.get("weight_decay", 0.01),
            "warmup_steps": config.get("warmup_steps", 100),
            "diffusion_steps": config.get("diffusion_steps", 6),
            "noise_schedule": config.get("noise_schedule", "cosine"),
            "dropout": config.get("dropout", 0.0),
            "mlp_expansion": config.get("mlp_expansion", 4),
            "mask_rate_min": config.get("mask_rate_min", 0.15),
            "mask_rate_max": config.get("mask_rate_max", 0.8),
            "leader_fraction": config.get("leader_fraction", 0.12),
            "death_streak": config.get("death_streak", 5),
            "refinement_steps": config.get("refinement_steps", 6),
            "topk": config.get("topk", 8),
        }
        log.info("Set run config params from config.json")
    else:
        # Fallback defaults
        run["config"] = {
            "hidden_dim": 256,
            "seq_len": 256,
            "batch_size": 1,
            "num_blocks": 4,
            "num_ants": 128,
            "vocab_size": 50257,
        }
        log.info("Set run config params from defaults (config.json not found)")

    state = load_curriculum_state()
    if state:
        run["curriculum"] = {
            "total_books": state.get("total_books", 203),
            "current_book": state.get("current_book", "unknown"),
            "book_index": state.get("book_index", 0),
            "plateau_window_secs": state.get("plateau_window_secs", 300),
            "plateau_threshold": state.get("plateau_threshold", 0.005),
            "hard_timeout_secs": state.get("hard_timeout_secs", 900),
            "startup_timeout_secs": state.get("startup_timeout_secs", 600),
            "switch_reason": state.get("switch_reason", "unknown"),
            "switch_time": state.get("switch_time", ""),
        }
        log.info("Set run curriculum params: book=%s index=%d",
                 state.get("current_book"), state.get("book_index", 0))

    global_step = load_step()
    if global_step > 0:
        run["global_step_at_start"] = global_step
        log.info("Global step at watcher start: %d", global_step)


def create_experiment_run(repo: str, mode: str) -> tuple[Run, str]:
    """Create or resume one Aim run per EXPERIMENT_DIR/exp_id."""
    experiment_id = infer_experiment_id()
    run_hash = build_run_hash(experiment_id)
    run_name = Path(EXPERIMENT_DIR).name

    try:
        run = Run(run_hash=run_hash, repo=repo, experiment=AIM_EXPERIMENT)
        log.info(
            "Resumed Aim run for exp_id=%s (name=%s hash=%s)",
            experiment_id,
            run_name,
            run.hash,
        )
    except Exception as e:
        # Aim can refuse stale softlocks. Fall back to a unique hash but keep same tags.
        if "softlock" in str(e).lower() or "file lock" in str(e).lower():
            ts_hash = hashlib.sha1(f"{run_hash}|{int(time.time())}".encode("utf-8")).hexdigest()[:24]
            run = Run(run_hash=ts_hash, repo=repo, experiment=AIM_EXPERIMENT)
            run["run_hash_fallback_from"] = run_hash
            try_add_tag(run, "lock-fallback")
            log.warning(
                "Lock conflict for hash=%s; created fallback run hash=%s",
                run_hash,
                ts_hash,
            )
        else:
            log.warning("Could not open deterministic run %s: %s. Creating new.", run_hash, e)
            run = Run(repo=repo, experiment=AIM_EXPERIMENT)

    run.name = run_name
    run["training_type"] = "diffusion"
    run["experiment_id"] = experiment_id
    run["experiment_dir"] = EXPERIMENT_DIR
    run["watcher_mode"] = mode
    run["metrics_file"] = METRICS_FILE

    try_add_tag(run, "erm")
    try_add_tag(run, "diffusion")
    try_add_tag(run, f"exp:{experiment_id}")
    try_add_tag(run, f"expdir:{run_name}")
    try_add_tag(run, f"mode:{mode}")

    set_run_params(run)
    return run, experiment_id


def track_metrics(
    run: Run,
    record: dict,
    current_book: str = "unknown",
    default_experiment_id: str = "unknown",
):
    """Track a single metrics record with context. Step is the global step counter."""
    step = record.get("step", 0)
    seq_len = record.get("seq_len", 256)
    loss = record.get("loss")
    edits = record.get("edits")
    experiment_id = record.get("exp_id", default_experiment_id)
    context = {"subset": "train", "book": current_book, "exp_id": experiment_id}

    for key in TRACKED_METRICS:
        if key in record:
            run.track(
                record[key],
                name=key,
                step=step,
                context=context,
            )

    # Loss-per-edit and edit-rate as accuracy proxies
    if loss is not None and edits is not None:
        if edits > 0:
            run.track(
                loss / edits,
                name="loss_per_edit",
                step=step,
                context=context,
            )
        if seq_len > 0:
            run.track(
                edits / seq_len,
                name="edit_rate",
                step=step,
                context=context,
            )


def track_ant_state(
    run: Run,
    step: int,
    current_book: str = "unknown",
    experiment_id: str = "unknown",
):
    """Read ant_state.json and track colony metrics."""
    ant_data = load_ant_state()
    if not ant_data:
        return

    ant_types = ant_data.get("ant_type", [])
    streaks = ant_data.get("streak", [])

    if ant_types:
        num_leaders = ant_types.count("Leader")
        num_followers = ant_types.count("Follower")
        total = len(ant_types)
        leader_frac = num_leaders / total if total > 0 else 0.0

        ctx = {"type": "ant_colony", "book": current_book, "exp_id": experiment_id}
        run.track(num_leaders, name="num_leaders", step=step, context=ctx)
        run.track(num_followers, name="num_followers", step=step, context=ctx)
        run.track(leader_frac, name="leader_fraction", step=step, context=ctx)
        run.track(total, name="total_ants", step=step, context=ctx)

    if streaks:
        mean_streak = sum(streaks) / len(streaks)
        max_streak = max(streaks)
        min_streak = min(streaks)
        zero_streak_count = streaks.count(0)
        ctx = {"type": "ant_colony", "book": current_book, "exp_id": experiment_id}
        run.track(mean_streak, name="mean_streak", step=step, context=ctx)
        run.track(float(max_streak), name="max_streak", step=step, context=ctx)
        run.track(zero_streak_count, name="zero_streak_ants", step=step, context=ctx)

        # Track streak distribution
        try:
            run.track(
                Distribution(np.array(streaks, dtype=np.float64)),
                name="streak_distribution",
                step=step,
                context={"type": "ant_colony", "book": current_book, "exp_id": experiment_id},
            )
        except Exception as e:
            log.debug("Could not track streak distribution: %s", e)


def track_io_examples(
    run: Run,
    filepath: str,
    current_book: str = "unknown",
    experiment_id: str = "unknown",
):
    """Read io_example.json and track them as Aim Text examples."""
    data = load_json_file(filepath)
    if not data or "examples" not in data:
        return

    metadata = data.get("metadata", {})
    step = metadata.get("checkpoint_step", 0)

    for ex in data.get("examples", []):
        example_idx = ex.get("example_idx", 0)
        clean = ex.get("clean_string", "")
        corrupted = ex.get("corrupted_string", "")
        predictions = ex.get("position_predictions", [])

        # Format a nice text block
        text_lines = [f"EXAMPLE {example_idx} (Step {step})"]
        text_lines.append(f"CLEAN:     {clean[:200]}...")
        text_lines.append(f"CORRUPTED: {corrupted[:200]}...")
        text_lines.append(f"CORRUPTED POSITIONS: {len(predictions)}")
        text_lines.append("")
        
        # Show up to 5 predictions
        for pred in predictions[:5]:
            pos = pred.get("position", "?")
            gt_text = pred.get("ground_truth_text", "?")
            corr_type = pred.get("corruption_type", "?")
            top3 = pred.get("top_3_predictions", [])
            
            top_preds_str = ", ".join([f"{p.get('text', '?')} ({p.get('p', '0')})" for p in top3])
            text_lines.append(f"Pos {pos} [{corr_type}] GT='{gt_text}' → Preds: {top_preds_str}")

        if len(predictions) > 5:
            text_lines.append(f"... and {len(predictions) - 5} more positions")

        full_text = "\n".join(text_lines)

        run.track(
            Text(full_text),
            name="io_examples",
            step=step,
            context={
                "subset": "train",
                "book": current_book,
                "exp_id": experiment_id,
                "example_idx": example_idx,
            },
        )
    log.info("Tracked %d IO examples from step %d", len(data.get("examples", [])), step)


def track_book_summary(run: Run, book_index: int, book_name: str,
                       loss_values: list, step_values: list):
    """Track per-book summary statistics when a book switch occurs."""
    if not loss_values:
        log.info("No loss values for book %d (%s), skipping summary", book_index, book_name)
        return

    ctx = {"type": "per_book", "book": book_name}
    loss_start = loss_values[0]
    loss_end = loss_values[-1]
    loss_min = min(loss_values)
    loss_delta = loss_end - loss_start  # Negative = improvement
    steps_on_book = len(loss_values)

    run.track(loss_delta, name="loss_delta", step=book_index, context=ctx)
    run.track(loss_start, name="loss_start", step=book_index, context=ctx)
    run.track(loss_end, name="loss_end", step=book_index, context=ctx)
    run.track(loss_min, name="loss_min", step=book_index, context=ctx)
    run.track(float(steps_on_book), name="steps_on_book", step=book_index, context=ctx)

    # Loss distribution for this book
    try:
        arr = np.array(loss_values, dtype=np.float64)
        run.track(
            Distribution(arr),
            name="loss_distribution",
            step=book_index,
            context={"book": book_name},
        )
        log.info(
            "Book %d (%s) summary: delta=%.4f, start=%.4f, end=%.4f, min=%.4f, steps=%d",
            book_index, book_name, loss_delta, loss_start, loss_end, loss_min, steps_on_book,
        )
    except Exception as e:
        log.warning("Could not track loss distribution for book %d: %s", book_index, e)


def log_book_switch(run: Run, book_index: int, state: dict):
    """Log a book switch as run hyperparameters and update curriculum params."""
    book_name = state.get("current_book", "unknown")
    loss = state.get("loss_at_switch", 0.0)
    reason = state.get("switch_reason", "unknown")

    run[f"book_{book_index}"] = book_name
    run[f"loss_at_book_{book_index}_switch"] = loss
    run[f"switch_reason_book_{book_index}"] = reason
    run["current_book_index"] = book_index
    run["current_book"] = book_name

    # Update curriculum params on each switch
    run["curriculum"] = {
        "total_books": state.get("total_books", 203),
        "current_book": book_name,
        "book_index": book_index,
        "plateau_window_secs": state.get("plateau_window_secs", 300),
        "plateau_threshold": state.get("plateau_threshold", 0.005),
        "hard_timeout_secs": state.get("hard_timeout_secs", 900),
        "startup_timeout_secs": state.get("startup_timeout_secs", 600),
        "switch_reason": reason,
        "switch_time": state.get("switch_time", ""),
    }

    # Log as Text event
    run.track(
        Text(f"Book switch → {book_name} (index={book_index}, reason={reason}, loss_at_switch={loss:.4f})"),
        name="training_events",
        step=book_index,
        context={"type": "book_switch"},
    )

    log.info("Book switch → index=%d, book=%s, loss=%.4f, reason=%s",
             book_index, book_name, loss, reason)


# ---------------------------------------------------------------------------
# Backfill mode
# ---------------------------------------------------------------------------

def backfill():
    """Read all historical metrics_book*.jsonl files and push to a single Aim run."""
    pattern = os.path.join(EXPERIMENT_DIR, "metrics_book*.jsonl")
    files = sorted(glob.glob(pattern))

    if not files:
        log.warning("No backup files found matching %s", pattern)
        return

    log.info("Backfilling %d historical files into single Aim run", len(files))

    run, experiment_id = create_experiment_run(AIM_REPO, mode="backfill")
    run["backfill"] = True

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        match = re.search(r"metrics_book(\d+)", filename)
        book_num = int(match.group(1)) if match else 0

        log.info("Backfilling %s (book %d)", filename, book_num)

        records = []
        with open(filepath, "r") as f:
            for line in f:
                rec = parse_line(line)
                if rec:
                    records.append(rec)

        if not records:
            log.warning("No valid records in %s, skipping", filename)
            continue

        # Extract book name from exp_id field
        exp_id = records[0].get("exp_id", f"book{book_num}")
        book_name = exp_id  # e.g. "curriculum-book0"
        run[f"book_{book_num}"] = exp_id
        run[f"source_file_book_{book_num}"] = filename

        # Track all records with context
        loss_values = []
        step_values = []
        for rec in records:
            track_metrics(
                run,
                rec,
                current_book=book_name,
                default_experiment_id=experiment_id,
            )
            if "loss" in rec:
                loss_values.append(rec["loss"])
            if "step" in rec:
                step_values.append(rec["step"])

        # Track per-book summary
        track_book_summary(run, book_num, book_name, loss_values, step_values)

        local_max = max(r.get("step", 0) for r in records)
        log.info("  → %d records, max step=%d", len(records), local_max)

    # Also backfill the current live metrics.jsonl if it exists
    if os.path.exists(METRICS_FILE):
        log.info("Backfilling current metrics.jsonl")
        state = load_curriculum_state()
        current_book = state.get("current_book", "unknown") if state else "unknown"
        with open(METRICS_FILE, "r") as f:
            for line in f:
                rec = parse_line(line)
                if rec:
                    track_metrics(
                        run,
                        rec,
                        current_book=current_book,
                        default_experiment_id=experiment_id,
                    )

    # Track ant state at backfill time
    global_step = load_step()
    track_ant_state(
        run,
        global_step,
        current_book="backfill",
        experiment_id=experiment_id,
    )

    run.close()
    log.info("Backfill complete — single run created with all historical data")


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch():
    """Tail metrics.jsonl and stream to Aim. Single run, detect book switches."""
    log.info("Starting watcher — file=%s, server=%s", METRICS_FILE, AIM_REPO)

    run, experiment_id = create_experiment_run(AIM_REPO, mode="watch")

    # Attach initial curriculum state
    last_book_index = -1
    current_book_name = "unknown"
    state = load_curriculum_state()
    if state:
        last_book_index = state.get("book_index", -1)
        current_book_name = state.get("current_book", "unknown")
        log_book_switch(run, last_book_index, state)
        # Attach static params
        for key in ["plateau_window_secs", "plateau_threshold", "hard_timeout_secs",
                     "startup_timeout_secs", "total_books"]:
            if key in state:
                run[key] = state[key]

    last_inode = None
    last_pos = 0
    last_curriculum_mtime = 0
    last_ant_state_mtime = 0
    last_io_example_mtime = 0
    IO_EXAMPLE_FILE = os.path.join(EXPERIMENT_DIR, "checkpoints", "latest", "io_example.json")

    # Per-book loss tracking for summaries
    book_loss_values = []
    book_step_values = []

    # Plateau detection
    recent_losses = []

    # Track ant state on startup
    try:
        track_ant_state(
            run,
            load_step(),
            current_book_name,
            experiment_id=experiment_id,
        )
        log.info("Tracked initial ant colony state")
    except Exception as e:
        log.warning("Could not track initial ant state: %s", e)

    # Wait for the file to exist
    while not os.path.exists(METRICS_FILE):
        log.info("Waiting for %s to appear...", METRICS_FILE)
        time.sleep(POLL_INTERVAL)

    while True:
        try:
            # Check for file rotation (inode change or file shrunk)
            try:
                stat = os.stat(METRICS_FILE)
                current_inode = stat.st_ino
                current_size = stat.st_size
            except FileNotFoundError:
                log.info("File disappeared, waiting for it to reappear...")
                time.sleep(POLL_INTERVAL)
                continue

            if last_inode is not None and (current_inode != last_inode or current_size < last_pos):
                log.info("File rotation detected (inode %s→%s, size %d→%d), resetting position",
                         last_inode, current_inode, last_pos, current_size)
                last_pos = 0
                # File rotation likely means new book — reset book accumulators
                book_loss_values = []
                book_step_values = []

            last_inode = current_inode

            # Read new lines
            if current_size > last_pos:
                with open(METRICS_FILE, "r") as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()

                for line in new_lines:
                    rec = parse_line(line)
                    if not rec:
                        continue

                    track_metrics(
                        run,
                        rec,
                        current_book=current_book_name,
                        default_experiment_id=experiment_id,
                    )

                    # Accumulate per-book stats
                    if "loss" in rec:
                        book_loss_values.append(rec["loss"])
                        recent_losses.append(rec["loss"])

                    if "step" in rec:
                        book_step_values.append(rec["step"])

                    # Plateau detection
                    if len(recent_losses) > PLATEAU_WINDOW:
                        recent_losses = recent_losses[-PLATEAU_WINDOW:]
                    if len(recent_losses) >= PLATEAU_WINDOW:
                        variance = max(recent_losses) - min(recent_losses)
                        if variance < PLATEAU_THRESHOLD:
                            step = rec.get("step", 0)
                            avg_loss = sum(recent_losses) / len(recent_losses)
                            run.track(
                                Text(
                                    f"Possible plateau at step {step}, "
                                    f"loss={avg_loss:.4f} (range={variance:.5f}), "
                                    f"book={current_book_name}"
                                ),
                                name="training_events",
                                step=step,
                                context={"type": "plateau"},
                            )
                            log.info(
                                "Plateau detected: step=%d, loss=%.4f, range=%.5f",
                                step, avg_loss, variance,
                            )
                            # Reset so we don't spam plateau events
                            recent_losses = []

            # Check curriculum_state.json for book switches
            try:
                cs_mtime = os.path.getmtime(CURRICULUM_STATE_FILE)
                if cs_mtime > last_curriculum_mtime:
                    last_curriculum_mtime = cs_mtime
                    state = load_curriculum_state()
                    if state:
                        new_book_index = state.get("book_index", -1)
                        if new_book_index != last_book_index:
                            # Track summary for the book we're leaving
                            if last_book_index >= 0 and book_loss_values:
                                track_book_summary(
                                    run, last_book_index, current_book_name,
                                    book_loss_values, book_step_values,
                                )

                            # Switch to new book
                            log_book_switch(run, new_book_index, state)
                            current_book_name = state.get("current_book", "unknown")
                            last_book_index = new_book_index

                            # Reset per-book accumulators
                            book_loss_values = []
                            book_step_values = []
                            recent_losses = []
            except FileNotFoundError:
                pass

            # Check ant_state.json periodically (when it changes)
            try:
                ant_mtime = os.path.getmtime(ANT_STATE_FILE)
                if ant_mtime > last_ant_state_mtime:
                    last_ant_state_mtime = ant_mtime
                    global_step = load_step()
                    track_ant_state(
                        run,
                        global_step,
                        current_book_name,
                        experiment_id=experiment_id,
                    )
            except FileNotFoundError:
                pass

            # Check io_example.json periodically (when it changes)
            try:
                io_mtime = os.path.getmtime(IO_EXAMPLE_FILE)
                if io_mtime > last_io_example_mtime:
                    last_io_example_mtime = io_mtime
                    track_io_examples(
                        run,
                        IO_EXAMPLE_FILE,
                        current_book_name,
                        experiment_id=experiment_id,
                    )
            except FileNotFoundError:
                pass

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log.info("Interrupted, closing run")
            # Track final book summary before closing
            if book_loss_values and last_book_index >= 0:
                track_book_summary(
                    run, last_book_index, current_book_name,
                    book_loss_values, book_step_values,
                )
            run.close()
            break
        except Exception as e:
            log.error("Error in watch loop: %s", e, exc_info=True)
            time.sleep(POLL_INTERVAL * 2)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "watch"

    if mode == "backfill":
        backfill()
    elif mode == "watch":
        watch()
    else:
        print(f"Unknown mode: {mode}. Use 'watch' or 'backfill'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
