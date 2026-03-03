# AIM Direct Reporting + Control Plan

## Goal

Move from a central watcher deployment to per-training-pod reporting so telemetry is emitted by the pod that owns the run, then add a control plane for launching/stopping jobs from AIM metadata.

## Phase 1: Direct Pod Reporting (Implemented)

- Add an `aim-sidecar` container to training Job manifests.
- Sidecar runs `scripts/aim-watcher-v2.py` with:
  - `AIM_EXPERIMENT=erm-training`
  - `EXPERIMENT_DIR` set to the pod's experiment output directory
  - `POLL_INTERVAL=1`
- Sidecar reads `metrics.jsonl` as it is written and tracks:
  - `loss`, `edits`, `mean_phi`, `deaths`
  - `lr`, `follower_temp`, `leader_temp`
- Sidecar tags runs with:
  - `exp:<exp_id>`
  - `expdir:<experiment_dir_name>`
  - `mode:watch`

Result: no external dependency for active run telemetry; run identity and metrics are attached at source.

## Phase 2: AIM-Driven Launch Queue (Next)

Implement a small launcher service (`scripts/aim-run-launcher.py`) that:

- polls AIM for runs tagged `status:queued`
- reads run params (hidden_dim, num_blocks, seq_len, etc.)
- renders/applies a Kubernetes Job from a template
- sets run tag `status:running` + `k8s_job:<job_name>`

This keeps experiment config source-of-truth in AIM while execution stays in Kubernetes.

## Phase 3: AIM-Driven Control (Next)

Add a control loop for active runs:

- watch for run params/tags like `control:stop` or `control:pause`
- map to Kubernetes action (`delete job`, `suspend`, etc.)
- write terminal tags (`status:stopped`, `stop_reason:<...>`) back to AIM

## Safety Notes

- AIM should not directly mutate live pod internals.
- All control actions should go through Kubernetes API for auditability.
- Keep run config immutable once job starts; changes create new runs/jobs.
