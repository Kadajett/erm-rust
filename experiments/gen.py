#!/usr/bin/env python3
"""Generate ERM experiment K8s Job manifests as JSON."""
import json
import os

EXPERIMENTS = [
    ("exp-a", {"seq_len": 512, "hidden_dim": 192, "num_heads": 6, "num_ants": 64, "emax": 8, "batch_size": 2}),
    ("exp-b", {"seq_len": 384, "hidden_dim": 256, "num_heads": 8, "num_ants": 128, "emax": 16, "batch_size": 2}),
    ("exp-c", {"seq_len": 256, "hidden_dim": 256, "num_heads": 8, "num_ants": 256, "emax": 16, "batch_size": 1}),
    ("exp-d", {"seq_len": 512, "hidden_dim": 160, "num_heads": 5, "num_ants": 96, "emax": 12, "batch_size": 2}),
    ("exp-e", {"seq_len": 384, "hidden_dim": 192, "num_heads": 6, "num_ants": 64, "emax": 16, "batch_size": 4}),
    ("exp-f", {"seq_len": 256, "hidden_dim": 128, "num_heads": 4, "num_ants": 128, "emax": 8, "batch_size": 4}),
]

BASE_CONFIG = {
    "vocab_size": 0, "num_blocks": 3, "mlp_expansion": 4, "dropout": 0.0,
    "topk": 6, "pmax": 6, "refinement_steps": 4,
    "mask_rate_max": 0.8, "mask_rate_min": 0.15,
    "replace_rate_max": 0.1, "replace_rate_min": 0.02,
    "pheromone_evap": 0.1, "pheromone_eta": 0.7,
    "taint_zeta": 0.3, "taint_max": 5.0, "taint_decay": 0.05,
    "phi_max": 100.0, "phi_init": 0.05,
    "route_epsilon": 1e-6, "route_lambda": 1.0, "route_mu": 0.01,
    "prune_min_score": -1.0, "prune_max_age": 1000,
    "leader_ema_gamma": 0.3, "death_streak": 5,
    "max_edits_per_step": 0.18, "leader_fraction": 0.12,
    "learning_rate": 0.0005, "weight_decay": 0.01, "warmup_steps": 100,
}

# Read the script template from a separate file
script_tpl_path = os.path.join(os.path.dirname(__file__), "train-script.sh.tpl")
with open(script_tpl_path) as f:
    SCRIPT_TPL = f.read()

def make_job(exp_id, params):
    config = {**BASE_CONFIG, **params}
    config_json = json.dumps(config, indent=2)
    variant = "seq{}-h{}-a{}-e{}-b{}".format(
        params['seq_len'], params['hidden_dim'], params['num_ants'],
        params['emax'], params['batch_size'])
    
    script = SCRIPT_TPL.replace("__EXP_ID__", exp_id).replace("__CONFIG_JSON__", config_json)
    
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "erm-" + exp_id,
            "namespace": "pcn-train",
            "labels": {"app": "erm-experiment", "exp": exp_id, "variant": variant},
        },
        "spec": {
            "backoffLimit": 1,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {"labels": {"app": "erm-experiment", "exp": exp_id}},
                "spec": {
                    "restartPolicy": "Never",
                    "runtimeClassName": "nvidia",
                    "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [{"matchExpressions": [
                            {"key": "kubernetes.io/hostname", "operator": "In", "values": ["theshire"]}
                        ]}]
                    }}},
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

items = [make_job(eid, p) for eid, p in EXPERIMENTS]
out = {"apiVersion": "v1", "kind": "List", "items": items}
out_path = os.path.join(os.path.dirname(__file__), "erm-experiments.json")
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print("Generated {} jobs to {}".format(len(items), out_path))
