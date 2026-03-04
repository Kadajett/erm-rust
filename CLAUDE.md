# ERM-Rust Project Rules

## Build & Deploy

- **Always compile with CUDA**: The k8s cluster node (theshire) has an NVIDIA RTX 3050 with CUDA 13.0. Always build with `--features cuda` (or `--features gpu,cuda`) and use `--backend cuda` in k8s job YAMLs.
- **Builder pod**: Use `k8s/erm-builder-pod.yaml` to compile. It syncs source from the workspace, builds with CUDA, and outputs to `/workspace/erm-rust/bin/erm.new`. Rename to `erm` after verifying.
- **Deploy flow**: Delete existing job → `kubectl apply -f k8s/erm-alice-run-b1.yaml` → verify pod starts and GPU is active with `nvidia-smi`.
- **Binary swap**: The running pod holds the binary file lock. Stop the training job first, then swap `erm.new` → `erm` via an ephemeral pod, then redeploy.

## GPU Backend Notes

- **CUDA backend** (`--backend cuda`): Requires burn-cuda (cudarc). The cudarc version shipped with burn 0.20 needs CUDA 13.x driver APIs. Driver must be up-to-date (580.x with all CUDA 13 symbols).
- **wgpu backend** (`--backend gpu`): Uses Vulkan. Requires NVIDIA Vulkan ICD (`libnvidia-gl-580` package) and `NVIDIA_DRIVER_CAPABILITIES=all` or manual Vulkan ICD setup in the container. Without proper Vulkan setup, silently falls back to CPU.
- **CPU backend** (`--backend cpu`): Always works but very slow for d=512 models. Use as fallback only.
- **Container GPU setup**: The nvidia-container-runtime with `compute,utility` only injects compute libs. For Vulkan, need `graphics` capability or manual lib mounts. The toolkit on theshire has a known bug with `graphics`/`all` capabilities ("invalid symlink specification --link").

## Critical: cudarc CUDA Version Pinning

The builder pod uses an `nvcc` shim that reports CUDA 13.0 to cudarc's build-time detection. This is necessary because:
- The host has CUDA 13.1 toolkit (with nvcc) but only a CUDA 13.0 driver (580.x)
- cudarc 0.18.2 uses `cuda-version-from-build-system` + `fallback-latest` features
- If cudarc detects 13.1, it enables `cuDevSmResourceSplit` which doesn't exist in the 13.0 driver → runtime panic
- The shim at `/tmp/cuda-shim/nvcc` in the builder forces detection to 13.0

**If you upgrade the NVIDIA driver to support CUDA 13.1+, remove the nvcc shim from `k8s/erm-builder-pod.yaml`.**

## K8s

- Namespace: `pcn-train`
- Node: `theshire` (RTX 3050, 6GB VRAM, driver 580.126.20)
- Training job label: `app=erm-diffusion`
- Builder pod name: `erm-builder`
- Host paths: `/home/kadajett/dev/erm-rust` (deploy), `/home/kadajett/dev/erm-rust-src` (build cache)

## Experiment Tracking Skill (Required)

- Use shared skill: `skills/experiment-tracking/SKILL.md`
- Track all config/code training decisions in `experiments/tracking/theory-tracker.json`
- Keep the tracker as a JSON array of objects
- Mark hypotheses with tag `theory`
- If user asks for a new experiment id, default to full restart (no `--resume`) unless user explicitly requests resume.
- Each entry must include:
  - `decision.model` (model that made the decision)
  - `result.reported_by_model` (model that measured/reported impact)
  - `evaluation.window_minutes_target` (default `60`) and `evaluation.window_minutes_actual` (override if crash/short run)
  - VRAM and loss trend metrics

## Shared Operator Memory (Read First)

- Before changing training jobs or data order, read `docs/OPERATOR_MEMORY.md`.
- Keep `docs/OPERATOR_MEMORY.md` current with:
  - live job + experiment id
  - dataset order for current and next staged run
  - absolute host and pod data paths
  - any critical helper commands used by the operator
- For plateau/debug analysis, follow the canonical workflow in `docs/OPERATOR_MEMORY.md`:
  - pull pod logs + metrics writer status
  - validate process + GPU state
  - copy `metrics.jsonl` locally via `kubectl cp`
  - compute fixed windows with `jq` + `awk`
  - compare windows across recent experiments before recommending changes
