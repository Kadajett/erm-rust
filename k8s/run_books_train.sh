#!/bin/bash
set -e
echo "=== ERM Books Aggressive Training (GPU) ==="
echo "Host: $(hostname)"

apt-get update -qq && apt-get install -y -qq libvulkan1 mesa-vulkan-drivers vulkan-tools ca-certificates curl ffmpeg librsvg2-bin 2>&1 | tail -n 5 || true

mkdir -p /etc/vulkan/icd.d
echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3"}}' > /etc/vulkan/icd.d/nvidia_icd.json || true

mkdir -p /workspace/erm-rust/data/text
if [ -d /workspace/data/books ]; then
  echo "Found books dir: /workspace/data/books"
  cat /workspace/data/books/*.txt > /workspace/erm-rust/data/text/corpus_books.txt || true
elif [ -d /workspace/data/books-code ]; then
  cat /workspace/data/books-code/*.txt > /workspace/erm-rust/data/text/corpus_books.txt || true
else
  echo "No books found under /workspace/data/books or /workspace/data/books-code" >&2
  exit 1
fi

echo "Corpus bytes: $(wc -c < /workspace/erm-rust/data/text/corpus_books.txt)"

cat > /workspace/erm-rust/data/books_aggressive_config.json <<'JSON'
{
  "vocab_size": 0,
  "seq_len": 1024,
  "hidden_dim": 384,
  "num_blocks": 4,
  "num_heads": 8,
  "mlp_expansion": 4,
  "dropout": 0.0,
  "emax": 32,
  "num_ants": 256,
  "topk": 8,
  "pmax": 8,
  "refinement_steps": 6,
  "batch_size": 2,
  "mask_rate_max": 0.8,
  "mask_rate_min": 0.15,
  "replace_rate_max": 0.1,
  "replace_rate_min": 0.02,
  "pheromone_evap": 0.1,
  "pheromone_eta": 0.8,
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
  "max_edits_per_step": 0.20,
  "leader_fraction": 0.15,
  "learning_rate": 0.0005,
  "weight_decay": 0.01,
  "warmup_steps": 100
}
JSON

echo "Starting GPU training (aggressive)"

/workspace/erm-rust/bin/erm colony-train \
  --data /workspace/erm-rust/data/text/corpus_books.txt \
  --steps 20000 \
  --config /workspace/erm-rust/data/books_aggressive_config.json \
  --backend gpu \
  --log-every 100 \
  --checkpoint-dir /workspace/erm-rust/data/colony-checkpoints 2>&1 | tee /workspace/erm-rust/data/colony-books-train-log.txt

echo "=== Training finished ==="
