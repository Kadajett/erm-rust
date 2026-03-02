# Migration: Char-Level → Token-Level Training

## Overview

This guide covers migrating existing ERM checkpoints and training runs from
the old char-level pipeline to the new BPE subword token-level pipeline.

---

## What Changed

### Tokenizer

| Before | After |
|--------|-------|
| `CharTokenizer` (one id per UTF-8 char) | `BpeTokenizer` (subword merges) |
| vocab_size ≈ 100–300 | vocab_size ≈ 4096–8192 |
| `erm_config.vocab_size` = num chars | `erm_config.vocab_size` = BPE vocab size |
| PAD=0, MASK=1, chars 2.. | PAD=0, MASK=1, UNK=2, subwords 3.. |

### Config Fields Added

```json
{
  "diffusion_steps": 6,
  "noise_schedule": "cosine",
  "gamma_min": 0.5,
  "gamma_max": 2.0,
  "use_paragraph_spans": true,
  "tokenizer_type": "bpe",
  "bpe_vocab_size": 4096,
  "bpe_vocab_path": "",
  "exp_id": "exp-a",
  "metrics_path": ""
}
```

All fields have defaults in `ErmConfig::default()` — old config files will
load and work (defaults applied for missing fields via `serde(default)`).

### Training Command

**Old (char-level, single file):**
```bash
erm colony-train \
  --data /workspace/corpus.txt \
  --steps 10000 \
  --backend gpu
```

**New (BPE, streaming directory, diffusion loss):**
```bash
erm diffusion-train \
  --data /workspace/rust-pcn/data/books \
  --steps 10000 \
  --backend gpu \
  --exp-id exp-a \
  --seq-len 512 \
  --hidden-dim 192 \
  --num-ants 64 \
  --batch 2 \
  --diffusion-t 6 \
  --checkpoint-dir /data/experiments/exp-a \
  --checkpoint-every 500 \
  --log-every 50
```

### Data Format

**Old:** single `.txt` file concatenated corpus.

**New:** directory of `.txt` files. Each file is tokenized on-the-fly
(CPU thread) and fed as `TokenBatch` items to the GPU trainer.

---

## Migrating Existing Checkpoints

Old checkpoints (scorer.bin from `colony-train`) are **not directly
compatible** because:

1. Embedding table size changed (char vocab → BPE vocab)
2. Config shape may differ

**Option A: Start fresh** (recommended for production)

```bash
erm diffusion-train --data <books-dir> --steps 10000 ...
```

**Option B: Use old checkpoint as warm-start only**

The scorer weights can be used as a warm-start if vocab sizes match.
If they differ, use `--config` to re-specify the vocab size and the scorer
will be re-initialized. Only the graph.json and ant_state.json transfer.

---

## Running Legacy Colony Train

The old `colony-train` command still works unchanged for backward compat:

```bash
erm colony-train --data corpus.txt --steps 5000 --backend gpu
```

It uses `CharTokenizer` and the original training loop. No diffusion steps.

---

## Checklist

- [ ] Replace `--data <file>` with `--data <directory>` for book data
- [ ] Add `--exp-id <name>` for metrics tracking
- [ ] Add `--checkpoint-dir <path>` with `--checkpoint-every 500`
- [ ] BPE vocab auto-trains from corpus on first run (or provide `bpe_vocab_path`)
- [ ] Verify `metrics.jsonl` is being written (check `--checkpoint-dir`)
- [ ] Verify no full-corpus GPU load: watch GPU memory during first 10 steps

---

## Expected Metrics Changes

With BPE tokenization:
- `loss` typically starts higher (larger vocab → harder CE) but converges faster
- `edits` per step = same range (colony budget unchanged)
- `seq_len` is effectively 5-10× more semantically dense than char-level

A BPE seq_len=512 covers roughly the same text as char-level seq_len=2048–4096.
