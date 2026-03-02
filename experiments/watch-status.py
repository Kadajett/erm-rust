#!/usr/bin/env python3
"""
STATUS.json watcher for ERM diffusion experiments.

Polls experiment log files and metrics.jsonl to build a current STATUS.json.
Run from the erm-rust root:

  python3 experiments/watch-status.py

Or as a one-shot update:

  python3 experiments/watch-status.py --once

Writes to experiments/STATUS.json.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENTS = [
    {"id": "exp-a", "seq_len": 1024, "hidden_dim": 192, "ants": 64, "batch": 1, "T": 6},
    {"id": "exp-b", "seq_len": 768,  "hidden_dim": 256, "ants": 96, "batch": 1, "T": 6},
    {"id": "exp-c", "seq_len": 512,  "hidden_dim": 256, "ants": 128, "batch": 2, "T": 8},
    {"id": "exp-d", "seq_len": 512,  "hidden_dim": 192, "ants": 192, "batch": 2, "T": 8},
    {"id": "exp-e", "seq_len": 384,  "hidden_dim": 160, "ants": 256, "batch": 4, "T": 8},
    {"id": "exp-f", "seq_len": 256,  "hidden_dim": 128, "ants": 384, "batch": 4, "T": 10},
]

# Paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
EXP_BASE = Path("/home/kadajett/dev/erm-rust/data/experiments")
STATUS_PATH = SCRIPT_DIR / "STATUS.json"
TARGET_STEPS = 10000


def read_last_metrics(exp_id: str) -> dict:
    """Read the last entry from metrics.jsonl for an experiment."""
    metrics_path = EXP_BASE / exp_id / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    try:
        last_line = None
        with open(metrics_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if last_line:
            return json.loads(last_line)
    except Exception as e:
        print(f"  [warn] {exp_id} metrics read error: {e}", file=sys.stderr)
    return {}


def read_last_log_line(exp_id: str) -> str:
    """Read the last meaningful line from train.log."""
    log_path = EXP_BASE / exp_id / "train.log"
    if not log_path.exists():
        return ""
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if line and "diffusion step" in line:
                return line
    except Exception:
        pass
    return ""


def parse_log_line(line: str) -> dict:
    """Parse a log line like '[diffusion step    100] loss=3.42 ...'"""
    result = {}
    m = re.search(r"\[diffusion step\s+(\d+)\]", line)
    if m:
        result["step"] = int(m.group(1))
    m = re.search(r"loss=([\d.]+)", line)
    if m:
        result["loss"] = float(m.group(1))
    m = re.search(r"edits=(\d+)", line)
    if m:
        result["edits"] = int(m.group(1))
    m = re.search(r"mean_φ=([\d.]+)", line)
    if m:
        result["mean_phi"] = float(m.group(1))
    return result


def get_pod_status(exp_id: str) -> str:
    """Get K8s pod status for this experiment via kubectl."""
    try:
        import subprocess
        result = subprocess.run(
            ["kubectl", "get", "jobs", f"erm-diff-{exp_id}",
             "-n", "pcn-train",
             "-o", "jsonpath={.status.conditions[0].type}"],
            capture_output=True, text=True, timeout=5
        )
        status = result.stdout.strip()
        if status == "Complete":
            return "Completed"
        if status == "Failed":
            return "Failed"
        # Check if pod is running
        result2 = subprocess.run(
            ["kubectl", "get", "pods",
             "-l", f"exp={exp_id}",
             "-n", "pcn-train",
             "-o", "jsonpath={.items[0].status.phase}"],
            capture_output=True, text=True, timeout=5
        )
        phase = result2.stdout.strip()
        if phase:
            return phase
        return "Pending"
    except Exception:
        return "Unknown"


def estimate_eta(step: int, rate: float) -> float | None:
    """Estimate minutes remaining to TARGET_STEPS."""
    if step is None or rate is None or rate <= 0:
        return None
    remaining = TARGET_STEPS - step
    if remaining <= 0:
        return 0.0
    return remaining / rate / 60.0


def build_status() -> dict:
    """Build the current STATUS dict."""
    experiments = []
    summary = {"running": 0, "pending": 0, "completed": 0, "failed": 0}

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        metrics = read_last_metrics(exp_id)
        log_line = read_last_log_line(exp_id)
        log_data = parse_log_line(log_line)
        pod_status = get_pod_status(exp_id)

        step = metrics.get("step") or log_data.get("step")
        loss = metrics.get("loss") or log_data.get("loss")
        edits = metrics.get("edits") or log_data.get("edits")
        mean_phi = metrics.get("mean_phi") or log_data.get("mean_phi")

        # Estimate rate from log timestamps (simplified: use first/last metrics).
        rate = None
        eta = None
        if step and step > 50:
            # Rough rate estimate from experiment timing.
            rate = None  # Would need timestamps in log for accurate estimate
            eta = None

        status_str = pod_status
        summary_key = {
            "Running": "running",
            "Pending": "pending",
            "Completed": "completed",
            "Failed": "failed",
        }.get(pod_status, "pending")
        summary[summary_key] = summary.get(summary_key, 0) + 1

        experiments.append({
            "exp": exp_id,
            "seq_len": exp["seq_len"],
            "hidden_dim": exp["hidden_dim"],
            "ants": exp["ants"],
            "batch": exp["batch"],
            "T": exp["T"],
            "status": status_str,
            "step": step,
            "loss": round(loss, 4) if loss else None,
            "edits": edits,
            "mean_phi": round(mean_phi, 4) if mean_phi else None,
            "rate_steps_sec": rate,
            "eta_minutes": round(eta, 1) if eta else None,
        })

    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_steps": TARGET_STEPS,
        "summary": summary,
        "experiments": experiments,
    }


def main():
    once = "--once" in sys.argv
    interval = 60  # seconds

    print(f"STATUS watcher started. Writing to: {STATUS_PATH}")
    if once:
        print("One-shot mode.")

    while True:
        try:
            status = build_status()
            with open(STATUS_PATH, "w") as f:
                json.dump(status, f, indent=2)
            print(
                f"[{status['timestamp']}] Updated STATUS.json "
                f"running={status['summary']['running']} "
                f"completed={status['summary']['completed']}"
            )
        except Exception as e:
            print(f"ERROR building status: {e}", file=sys.stderr)

        if once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
