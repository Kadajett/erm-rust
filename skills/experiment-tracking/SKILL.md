# Experiment Tracking Skill

Use this skill for any training experiment decision that changes config, code, runtime, or deployment behavior.

## Goal

Keep a single machine-readable ledger of:
- the decision/hypothesis,
- who made it,
- what changed,
- and measured impact over time.

## Source of Truth

- Tracker file: `experiments/tracking/theory-tracker.json`
- Format: JSON array (`[]`) of decision records.

## Required Record Fields

Every record must include:
- `id`: unique stable string (`<exp_id>-<utc>-<slug>`)
- `created_at_utc`
- `experiment_id`
- `tags`: include `"theory"` for hypotheses
- `decision`
  - `model`: model that made the decision (example: `gpt-5-codex`, `claude-opus-4-6`)
  - `type`: `config_change` | `code_change` | `k8s_change` | `analysis`
  - `summary`
  - `changes`: list of changed fields/files and from/to values
- `evaluation`
  - `window_minutes_target`: default `60`
  - `window_minutes_actual`: actual observed duration
  - `status`: `proposed` | `running` | `completed` | `crashed` | `rolled_back`
  - `stop_reason`
- `result`
  - `reported_by_model`: model that reported measured effects
  - `step_window`
  - `metrics`:
    - `loss`: include at least `start_mean`, `end_mean`, `min`, `max`
    - `vram_mb`: include at least `total`, `used_min`, `used_max`
    - `host_ram_bytes` (if available)
    - `oom`: boolean
  - `notes`

## Workflow Standard

1. Before applying a change:
   - append a new entry with `tags` containing `"theory"`, `evaluation.status="proposed"`.
2. After deploying/running:
   - update the same entry with measured numbers.
   - set `window_minutes_actual` and `evaluation.status` (`completed`/`crashed`).
3. If crash happens early:
   - keep `window_minutes_target=60`,
   - set `window_minutes_actual` to observed duration,
   - set `stop_reason` (OOM, panic, manual stop, etc.).
4. Keep metrics numeric and comparable across runs.

## JSON Discipline

- Never switch away from JSON array root.
- Do not store comments in JSON.
- Keep numbers as numbers (not strings), except IDs/timestamps.
