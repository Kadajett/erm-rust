# Diffusion Training Milestones (RTX 3050 6GB)

Baseline run (`alice-run-b2`) ended at step `10,000`.

## Math

- Tokens per step: `batch_size * seq_len = 1 * 256 = 256`
- Token exposure at step `S`: `256 * S`

Current exposure:
- `10,000 * 256 = 2.56M` tokens

Milestone exposures:
- `100k` total steps: `25.6M` tokens
- `400k` total steps: `102.4M` tokens
- `1M` total steps: `256M` tokens

## Increment Plan From 10k

1. Milestone A:
   - Target total: `100k`
   - Additional steps: `90k`
   - Rough ETA: `~1-2 days` (depends on backend/runtime stability)

2. Milestone B:
   - Target total: `400k`
   - Additional steps after A: `300k`
   - Rough ETA: `~4-7 days`

3. Milestone C:
   - Target total: `1M`
   - Additional steps after B: `600k`
   - Rough ETA: `~10-15 days`

## Commands

Use the helper script to resume from `latest` and keep writing checkpoints:

```bash
scripts/run-diffusion-milestone.sh \
  --checkpoint-dir /home/kadajett/dev/erm-rust/data/experiments/alice-run-b2/checkpoints \
  --data /home/kadajett/dev/rust-pcn/data/books \
  --exp-id alice-run-b2 \
  --add-steps 90000 \
  --backend cuda
```

Then:

```bash
scripts/run-diffusion-milestone.sh \
  --checkpoint-dir /home/kadajett/dev/erm-rust/data/experiments/alice-run-b2/checkpoints \
  --data /home/kadajett/dev/rust-pcn/data/books \
  --exp-id alice-run-b2 \
  --add-steps 300000 \
  --backend cuda
```

Then:

```bash
scripts/run-diffusion-milestone.sh \
  --checkpoint-dir /home/kadajett/dev/erm-rust/data/experiments/alice-run-b2/checkpoints \
  --data /home/kadajett/dev/rust-pcn/data/books \
  --exp-id alice-run-b2 \
  --add-steps 600000 \
  --backend cuda
```

## Notes

- Keep tokenizer/config fixed while measuring quality changes.
- Keep `checkpoint_every` small enough for recovery (`250` is fine).
- Use `samples.jsonl` + `metrics.jsonl` together to detect plateau/regression.
