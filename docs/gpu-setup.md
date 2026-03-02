# GPU Setup — ERM on TheShire K8s Cluster

## Hardware
- RTX 3050 (8GB VRAM) on node `theshire`
- Shared with PCN training pod (~4GB VRAM) — ERM gets ~2GB

## What Works (v3 pod)

### Key findings:
1. **burn uses wgpu/Vulkan**, not CUDA directly — need Vulkan ICD in container
2. **GPU time-slicing** enabled via NVIDIA device plugin ConfigMap (`replicas: 4`)
3. **Do NOT use `runtimeClassName: nvidia`** — causes "invalid symlink" errors with time-slicing. Just request `nvidia.com/gpu: "1"` in resources.
4. **Use `ubuntu:24.04`** as base image, install Vulkan at startup
5. **`#![recursion_limit = "512"]`** needed in erm-cli for wgpu type depth

### Container setup (in pod args):
```bash
apt-get update -qq && apt-get install -y -qq libvulkan1 mesa-vulkan-drivers vulkan-tools
mkdir -p /etc/vulkan/icd.d
echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3"}}' > /etc/vulkan/icd.d/nvidia_icd.json
```

### Required env vars:
```yaml
env:
- name: LD_LIBRARY_PATH
  value: /usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda/lib64
- name: NVIDIA_DRIVER_CAPABILITIES
  value: all
- name: NVIDIA_VISIBLE_DEVICES
  value: all
- name: WGPU_BACKEND
  value: vulkan
```

### CUDA mounts (for future direct CUDA use):
```yaml
volumeMounts:
- name: cuda-13-0
  mountPath: /usr/local/cuda-13.0
  readOnly: true
- name: cuda-13-1
  mountPath: /usr/local/cuda
  readOnly: true
volumes:
- name: cuda-13-0
  hostPath:
    path: /usr/local/cuda-13.0
- name: cuda-13-1
  hostPath:
    path: /usr/local/cuda-13.1
```

### Node affinity (required):
```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - theshire
```

## GPU Time-Slicing Config
Applied to NVIDIA device plugin:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvidia-device-plugin-config
  namespace: kube-system
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4
```

This makes the node report 4 allocatable GPUs (all time-sliced on the same physical GPU).

## VRAM Budget
- PCN sprite training: ~4GB
- ERM burn training: ~1-2GB (d=256, V=941, B=8, L=128)
- Total: ~5-6GB / 8GB available
- Monitor: `kubectl exec <pod> -n pcn-train -- nvidia-smi`

## Performance
- Step 100 in ~2 min (GPU via wgpu/Vulkan)
- Loss: 6.79 → 3.86 in first 100 steps
