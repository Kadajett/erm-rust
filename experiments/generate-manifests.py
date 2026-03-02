#!/usr/bin/env python3
"""Generate ERM experiment K8s Job manifests as YAML using json module."""
import json
import textwrap

EXPERIMENTS = [
    ("exp-a", {"seq_len": 512, "hidden_dim": 192, "num_heads": 6, "num_ants": 64, "emax": 8, "batch_size": 2}),
    ("exp-b", {"seq_len": 384, "hidden_dim": 256, "num_heads": 8, "num_ants": 128, "emax": 16, "batch_size": 2}),
    ("exp-c", {"seq_len": 256, "hidden_dim": 256, "num_heads": 8, "num_ants": 256, "emax": 16, "batch_size": 1}),
    ("exp-d", {"seq_len": 512, "hidden_dim": 160, "num_heads": 5, "num_ants": 96, "emax": 12, "batch_size": 2}),
    ("exp-e", {"seq_len": 384, "hidden_dim": 192, "num_heads": 6, "num_ants": 64, "emax": 16, "batch_size": 4}),
    ("exp-f", {"seq_len": 256, "hidden_dim": 128, "num_heads": 4, "num_ants": 128, "emax": 8, "batch_size": 4}),
]

BASE_CONFIG = {
    "vocab_size": 0,
    "num_blocks": 3,
    "mlp_expansion": 4,
    "dropout": 0.0,
    "topk": 6,
    "pmax": 6,
    "refinement_steps": 4,
    "mask_rate_max": 0.8,
    "mask_rate_min": 0.15,
    "replace_rate_max": 0.1,
    "replace_rate_min": 0.02,
    "pheromone_evap": 0.1,
    "pheromone_eta": 0.7,
    "taint_zeta": 0.3,
    "taint_max": 5.0,
    "taint_decay": 0.05,
    "phi_max": 100.0,
    "phi_init": 0.05,
    "route_epsilon": 1e-6,
    "route_lambda": 1.0,
    "route_mu": 0.01,
    "prune_min_score": -1.0,
    "prune_max_age": 1000,
    "leader_ema_gamma": 0.3,
    "death_streak": 5,
    "max_edits_per_step": 0.18,
    "leader_fraction": 0.12,
    "learning_rate": 0.0005,
    "weight_decay": 0.01,
    "warmup_steps": 100,
}

def make_script(exp_id, config):
    config_json = json.dumps(config, indent=2)
    return f'''set -e
EXP_ID="{exp_id}"
echo "=== ERM Experiment: $EXP_ID ==="
echo "Host: $(hostname) | Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

apt-get update -qq 2>&1 | tail -1 || true
apt-get install -y -qq --no-install-recommends libvulkan1 mesa-vulkan-drivers ca-certificates 2>&1 | tail -3 || true
mkdir -p /etc/vulkan/icd.d
echo '{{"file_format_version":"1.0.0","ICD":{{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}}}' > /etc/vulkan/icd.d/nvidia_icd.json || true

BOOK_DIR="/workspace/rust-pcn/data/books"
CORPUS="/workspace/erm-rust/data/text/corpus_books.txt"
mkdir -p /workspace/erm-rust/data/text
if [ ! -f "$CORPUS" ] || [ $(wc -c < "$CORPUS" 2>/dev/null || echo 0) -lt 1000 ]; then
  ls "$BOOK_DIR"/*.txt | head -80 | xargs cat > "$CORPUS" 2>/dev/null || true
fi

EXP_DIR="/workspace/erm-rust/data/experiments/$EXP_ID"
CKPT_DIR="$EXP_DIR/checkpoints"
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

cat > "$EXP_DIR/config.json" <<'JSONEOF'
{config_json}
JSONEOF

WARMSTART_DIR="/workspace/erm-rust/data/colony-checkpoints/warmstart"
WARMSTART_FLAG=""
if [ -d "$WARMSTART_DIR" ] && [ -f "$WARMSTART_DIR/graph.json" ]; then
  echo "Using warmstart from $WARMSTART_DIR"
  WARMSTART_FLAG="--warmstart $WARMSTART_DIR"
fi

echo "Config:"
cat "$EXP_DIR/config.json"
echo ""
echo "=== Starting $EXP_ID training at $(date -u +%H:%M:%S) ==="
echo ""

/workspace/erm-rust/bin/erm colony-train \\
  --data "$CORPUS" \\
  --steps 10000 \\
  --config "$EXP_DIR/config.json" \\
  --backend gpu \\
  --log-every 50 \\
  --checkpoint-dir "$CKPT_DIR" \\
  $WARMSTART_FLAG \\
  2>&1 | tee "$LOG_DIR/colony-train.log"

EXIT_CODE=$?
echo ""
echo "=== $EXP_ID finished (exit=$EXIT_CODE) at $(date -u +%H:%M:%S) ==="

if [ $EXIT_CODE -eq 0 ]; then
  WS="$EXP_DIR/warmstart"
  mkdir -p "$WS"
  cp "$CKPT_DIR/graph.json" "$WS/" 2>/dev/null || true
  cp "$CKPT_DIR/ant_state.json" "$WS/" 2>/dev/null || true
  cp "$EXP_DIR/config.json" "$WS/" 2>/dev/null || true
  echo "Warmstart saved to $WS"
fi
exit $EXIT_CODE
'''

def make_job_yaml(exp_id, params):
    config = {**BASE_CONFIG, **params}
    variant = f"seq{params['seq_len']}-h{params['hidden_dim']}-a{params['num_ants']}-e{params['emax']}-b{params['batch_size']}"
    script = make_script(exp_id, config)
    
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"erm-{exp_id}",
            "namespace": "pcn-train",
            "labels": {
                "app": "erm-experiment",
                "exp": exp_id,
                "variant": variant,
            },
        },
        "spec": {
            "backoffLimit": 1,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "erm-experiment",
                        "exp": exp_id,
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "runtimeClassName": "nvidia",
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [{
                                    "matchExpressions": [{
                                        "key": "kubernetes.io/hostname",
                                        "operator": "In",
                                        "values": ["theshire"],
                                    }]
                                }]
                            }
                        }
                    },
                    "containers": [{
                        "name": "erm-train",
                        "image": "ubuntu:24.04",
                        "command": ["/bin/bash", "-c"],
                        "args": [script],
                        "env": [
                            {"name": "LD_LIBRARY_PATH", "value": "/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"},
                            {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
                            {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
                            {"name": "WGPU_BACKEND", "value": "vulkan"},
                            {"name": "RUST_BACKTRACE", "value": "1"},
                            {"name": "RAYON_NUM_THREADS", "value": "2"},
                        ],
                        "resources": {
                            "requests": {"cpu": "1", "memory": "1Gi", "nvidia.com/gpu": "1"},
                            "limits": {"cpu": "4", "memory": "20Gi", "nvidia.com/gpu": "1"},
                        },
                        "volumeMounts": [
                            {"name": "erm-rust", "mountPath": "/workspace/erm-rust"},
                            {"name": "rust-pcn", "mountPath": "/workspace/rust-pcn", "readOnly": True},
                            {"name": "cuda-13-0", "mountPath": "/usr/local/cuda-13.0", "readOnly": True},
                            {"name": "cuda-13-1", "mountPath": "/usr/local/cuda", "readOnly": True},
                        ],
                    }],
                    "volumes": [
                        {"name": "erm-rust", "hostPath": {"path": "/home/kadajett/dev/erm-rust", "type": "Directory"}},
                        {"name": "rust-pcn", "hostPath": {"path": "/home/kadajett/dev/rust-pcn", "type": "Directory"}},
                        {"name": "cuda-13-0", "hostPath": {"path": "/usr/local/cuda-13.0"}},
                        {"name": "cuda-13-1", "hostPath": {"path": "/usr/local/cuda-13.1"}},
                    ],
                },
            },
        },
    }
    return json.dumps(job)

out_path = "/home/node/.openclaw/workspace/erm-rust/experiments/erm-experiments.yaml"
# Write as JSON list, then convert
# Actually, kubectl accepts JSON too. Let's write individual JSON files and use kubectl apply
# Or better: write the YAML manually since we can't import pyyaml

# Let's just use kubectl create with --from-file or JSON directly
jobs_json = []
for exp_id, params in EXPERIMENTS:
    jobs_json.append(make_job_yaml(exp_id, params))

# Write as a JSON list wrapped in a List kind
job_list = {
    "apiVersion": "v1",
    "kind": "List",
    "items": [json.loads(j) for j in jobs_json]
}

with open(out_path.replace('.yaml', '.json'), 'w') as f:
    json.dump(job_list, f, indent=2)

print(f"Generated {len(jobs_json)} Job manifests to {out_path.replace('.yaml', '.json')}")
