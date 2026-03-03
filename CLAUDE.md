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
