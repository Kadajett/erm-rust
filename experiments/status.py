#!/usr/bin/env python3
"""Generate STATUS.json for ERM experiments."""
import json
import subprocess
import re
import time
from datetime import datetime, timezone

EXPS = ["exp-a", "exp-b", "exp-c", "exp-d", "exp-e", "exp-f"]
TARGET_STEPS = 10000

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
    return r.stdout.strip()

def parse_step_line(line):
    """Parse: [colony step   299] loss=3.8897 edits=130 mean_phi=0.0430 ..."""
    m = re.search(r'step\s+(\d+)\].*?loss=([\d.]+).*?edits=(\d+).*?mean_.*?=([\d.]+)', line)
    if m:
        return int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4))
    return None

results = []
for exp in EXPS:
    pod = run("kubectl get pods -n pcn-train -l exp={} -o jsonpath='{{.items[0].metadata.name}}'".format(exp))
    status = run("kubectl get pods -n pcn-train -l exp={} -o jsonpath='{{.items[0].status.phase}}'".format(exp))
    
    entry = {
        "exp": exp,
        "pod": pod or "none",
        "status": status or "Unknown",
        "step": 0,
        "loss": None,
        "edits": None,
        "mean_phi": None,
        "rate_steps_sec": None,
        "eta_minutes": None,
    }
    
    if status in ("Running", "Succeeded"):
        # Get last step line
        logs = run("kubectl logs {} -n pcn-train --tail=5 2>/dev/null".format(pod))
        lines = [l for l in logs.split("\n") if "colony step" in l]
        if lines:
            parsed = parse_step_line(lines[-1])
            if parsed:
                entry["step"] = parsed[0]
                entry["loss"] = parsed[1]
                entry["edits"] = parsed[2]
                entry["mean_phi"] = parsed[3]
        
        # Estimate rate from start time + current step
        start_log = run("kubectl logs {} -n pcn-train 2>/dev/null | grep 'Starting.*training at' | head -1".format(pod))
        time_match = re.search(r'(\d{2}:\d{2}:\d{2})', start_log)
        if time_match and entry["step"] > 0:
            start_str = time_match.group(1)
            now = datetime.now(timezone.utc)
            start_h, start_m, start_s = map(int, start_str.split(":"))
            start_dt = now.replace(hour=start_h, minute=start_m, second=start_s, microsecond=0)
            if start_dt > now:
                start_dt = start_dt.replace(day=start_dt.day - 1)
            elapsed = (now - start_dt).total_seconds()
            if elapsed > 0:
                rate = entry["step"] / elapsed
                entry["rate_steps_sec"] = round(rate, 2)
                remaining = TARGET_STEPS - entry["step"]
                if rate > 0:
                    entry["eta_minutes"] = round(remaining / rate / 60, 1)
    
    results.append(entry)

status_doc = {
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "target_steps": TARGET_STEPS,
    "summary": {
        "running": sum(1 for r in results if r["status"] == "Running"),
        "pending": sum(1 for r in results if r["status"] == "Pending"),
        "completed": sum(1 for r in results if r["status"] == "Succeeded"),
        "failed": sum(1 for r in results if r["status"] not in ("Running", "Pending", "Succeeded", "Unknown")),
    },
    "experiments": results,
}

out_path = "/home/node/.openclaw/workspace/erm-rust/experiments/STATUS.json"
with open(out_path, "w") as f:
    json.dump(status_doc, f, indent=2)

print(json.dumps(status_doc, indent=2))
