#!/bin/bash
# Generate ERM experiment Job manifests
# Each experiment gets its own config, output dir, and K8s Job

OUT="/home/node/.openclaw/workspace/erm-rust/experiments/erm-experiments.yaml"
> "$OUT"

# Experiment definitions: EXP_ID SEQ_LEN HIDDEN_DIM NUM_HEADS NUM_ANTS EMAX BATCH_SIZE
declare -A EXPS
EXPS[exp-a]="512 192 6 64 8 2"
EXPS[exp-b]="384 256 8 128 16 2"
EXPS[exp-c]="256 256 8 256 16 1"
EXPS[exp-d]="512 160 5 96 12 2"
EXPS[exp-e]="384 192 6 64 16 4"
EXPS[exp-f]="256 128 4 128 8 4"

for EXP_ID in exp-a exp-b exp-c exp-d exp-e exp-f; do
  read SEQ_LEN HIDDEN_DIM NUM_HEADS NUM_ANTS EMAX BATCH_SIZE <<< "${EXPS[$EXP_ID]}"
  
  VARIANT="seq${SEQ_LEN}-h${HIDDEN_DIM}-a${NUM_ANTS}-e${EMAX}-b${BATCH_SIZE}"
  
  cat >> "$OUT" <<YAMLEOF
---
apiVersion: batch/v1
kind: Job
metadata:
  name: erm-${EXP_ID}
  namespace: pcn-train
  labels:
    app: erm-experiment
    exp: ${EXP_ID}
    variant: "${VARIANT}"
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: erm-experiment
        exp: ${EXP_ID}
    spec:
      restartPolicy: Never
      runtimeClassName: nvidia
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: In
                values: ["theshire"]
      containers:
      - name: erm-train
        image: ubuntu:24.04
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -e
          EXP_ID="${EXP_ID}"
          echo "=== ERM Experiment: \$EXP_ID ==="
          echo "Host: \$(hostname) | Start: \$(date -u +%Y-%m-%dT%H:%M:%SZ)"

          apt-get update -qq 2>&1 | tail -1 || true
          apt-get install -y -qq --no-install-recommends libvulkan1 mesa-vulkan-drivers ca-certificates 2>&1 | tail -3 || true
          mkdir -p /etc/vulkan/icd.d
          echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}' > /etc/vulkan/icd.d/nvidia_icd.json || true

          BOOK_DIR="/workspace/rust-pcn/data/books"
          CORPUS="/workspace/erm-rust/data/text/corpus_books.txt"
          mkdir -p /workspace/erm-rust/data/text
          if [ ! -f "\$CORPUS" ] || [ \$(wc -c < "\$CORPUS" 2>/dev/null || echo 0) -lt 1000 ]; then
            ls "\$BOOK_DIR"/*.txt | head -80 | xargs cat > "\$CORPUS" 2>/dev/null || true
          fi

          EXP_DIR="/workspace/erm-rust/data/experiments/\$EXP_ID"
          CKPT_DIR="\$EXP_DIR/checkpoints"
          LOG_DIR="\$EXP_DIR/logs"
          mkdir -p "\$CKPT_DIR" "\$LOG_DIR"

          cat > "\$EXP_DIR/config.json" <<'JSONEOF'
          {
            "vocab_size": 0,
            "seq_len": ${SEQ_LEN},
            "hidden_dim": ${HIDDEN_DIM},
            "num_blocks": 3,
            "num_heads": ${NUM_HEADS},
            "mlp_expansion": 4,
            "dropout": 0.0,
            "emax": ${EMAX},
            "num_ants": ${NUM_ANTS},
            "topk": 6,
            "pmax": 6,
            "refinement_steps": 4,
            "batch_size": ${BATCH_SIZE},
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
            "warmup_steps": 100
          }
JSONEOF

          WARMSTART_DIR="/workspace/erm-rust/data/colony-checkpoints/warmstart"
          WARMSTART_FLAG=""
          if [ -d "\$WARMSTART_DIR" ] && [ -f "\$WARMSTART_DIR/graph.json" ]; then
            WARMSTART_FLAG="--warmstart \$WARMSTART_DIR"
          fi

          echo "Config:"
          cat "\$EXP_DIR/config.json"
          echo ""
          echo "=== Starting \$EXP_ID training at \$(date -u +%H:%M:%S) ==="

          /workspace/erm-rust/bin/erm colony-train \\
            --data "\$CORPUS" \\
            --steps 10000 \\
            --config "\$EXP_DIR/config.json" \\
            --backend gpu \\
            --log-every 50 \\
            --checkpoint-dir "\$CKPT_DIR" \\
            \$WARMSTART_FLAG \\
            2>&1 | tee "\$LOG_DIR/colony-train.log"

          EXIT_CODE=\$?
          echo "=== \$EXP_ID finished (exit=\$EXIT_CODE) at \$(date -u +%H:%M:%S) ==="

          if [ \$EXIT_CODE -eq 0 ]; then
            WS="\$EXP_DIR/warmstart"
            mkdir -p "\$WS"
            cp "\$CKPT_DIR/graph.json" "\$WS/" 2>/dev/null || true
            cp "\$CKPT_DIR/ant_state.json" "\$WS/" 2>/dev/null || true
            cp "\$EXP_DIR/config.json" "\$WS/" 2>/dev/null || true
            echo "Warmstart saved to \$WS"
          fi
          exit \$EXIT_CODE
        env:
        - name: LD_LIBRARY_PATH
          value: "/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility"
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: WGPU_BACKEND
          value: "vulkan"
        - name: RUST_BACKTRACE
          value: "1"
        - name: RAYON_NUM_THREADS
          value: "2"
        resources:
          requests:
            cpu: "1"
            memory: "1Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "4"
            memory: "20Gi"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: erm-rust
          mountPath: /workspace/erm-rust
        - name: rust-pcn
          mountPath: /workspace/rust-pcn
          readOnly: true
        - name: cuda-13-0
          mountPath: /usr/local/cuda-13.0
          readOnly: true
        - name: cuda-13-1
          mountPath: /usr/local/cuda
          readOnly: true
      volumes:
      - name: erm-rust
        hostPath:
          path: /home/kadajett/dev/erm-rust
          type: Directory
      - name: rust-pcn
        hostPath:
          path: /home/kadajett/dev/rust-pcn
          type: Directory
      - name: cuda-13-0
        hostPath:
          path: /usr/local/cuda-13.0
      - name: cuda-13-1
        hostPath:
          path: /usr/local/cuda-13.1
YAMLEOF

done

echo "Generated $OUT"
wc -l "$OUT"
