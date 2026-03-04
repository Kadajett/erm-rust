# Operator Memory (Shared: Codex + Claude)

Last updated: 2026-03-04 UTC

## Current Live Run

- Job: `erm-alice-run-m1m-v7-sharded-3phase`
- Experiment id: `alice-run-b2-m1m-v7-sharded-3phase-r1`
- Status: running
- Confirmed phase/data order:
  - Phase 1: `100000` steps on `/workspace/rust-pcn/data/english-frontload-sharded`
  - Phase 2: `200000` steps on `/workspace/rust-pcn/data/sentence-bridge-smclm-sharded`
  - Phase 3: `700000` steps on `/workspace/rust-pcn/data/books`

Observed early-phase behavior:
- Entered classic early "the-stage" collapse in predictions (high-frequency stopword loops).
- Around steps `~2k-2.5k`, loss hovered in `~7.0-7.9` band with noisy but slight net downtrend.

### Startup Latency Note

- v4 startup was delayed by monolithic-file tokenization before first step emit.
- Mitigation added:
  - sharded frontload and bridge corpora
  - incremental line-stream tokenization in `StreamingDataset`
  - mid-epoch discovery of newly added `*.txt` files (no restart required after next rebuild/deploy)

## Canonical New-Experiment Deploy Flow (Fresh Start)

Use this exact sequence whenever user asks for a new experiment id and expects full restart.

1. Prepare a new job manifest with a new job name and new `NEW_EXP`.
   - Example pattern:
     - job name: `erm-alice-run-m1m-v6-sharded-3phase`
     - experiment id: `alice-run-b2-m1m-v6-sharded-3phase-r1`
   - Ensure script enforces scratch semantics:
     - `if [[ -d "${EXP_DIR}" ]]; then exit 1; fi`
     - first phase has no `--resume`
2. Stop old training job before rebuilding.
   - `kubectl delete job <old-job> -n pcn-train --ignore-not-found=true`
   - Wait for old pod termination.
3. Rebuild `erm.new` via builder pod.
   - `kubectl delete pod erm-builder -n pcn-train --ignore-not-found=true`
   - `kubectl apply -f k8s/erm-builder-pod.yaml`
   - wait until `Succeeded`
4. Verify binary publish succeeded.
   - Check builder logs for absence of:
     - `cp: cannot create regular file '/workspace/erm-rust/bin/erm.new': Text file busy`
   - Verify new artifact:
     - `ls -l /home/kadajett/dev/erm-rust/bin/erm.new`
     - `sha256sum /home/kadajett/dev/erm-rust/bin/erm.new`
5. Deploy new job.
   - `kubectl apply -f <new-job-yaml>`
6. Verify fresh start and correct experiment id from logs.
   - Must see:
     - `Scratch start: no checkpoint resume`
     - `DiffusionTrain: exp=<new-exp-id> ...`
   - Must not see `Resumed from checkpoint` in phase 1.
7. Confirm only new job is running.
   - `kubectl get jobs -n pcn-train -o wide`
   - `kubectl get pods -n pcn-train -o wide`

Notes:
- If build log ends with `Text file busy`, old trainer still had `erm.new` open. Kill old job and rebuild again.
- Training pods execute `/workspace/erm-rust/bin/erm.new`; rebuild must complete after old job is stopped.

## Dataset Paths (Host + Pod)

- High-quality English (HF `agentlans/high-quality-english-sentences`)
  - Host: `/home/kadajett/dev/rust-pcn/data/hq-english-sentences/train.txt`
  - Pod: `/workspace/rust-pcn/data/hq-english-sentences/train.txt`
  - Frontload dirs:
    - Host: `/home/kadajett/dev/rust-pcn/data/english-frontload`
    - Pod: `/workspace/rust-pcn/data/english-frontload`
  - Sharded frontload dirs (current v5):
    - Host: `/home/kadajett/dev/rust-pcn/data/english-frontload-sharded`
    - Pod: `/workspace/rust-pcn/data/english-frontload-sharded`

- Sentence bridge (HF `mmichall/smclm-10M-sentence-corpus`)
  - Host: `/home/kadajett/dev/rust-pcn/data/smclm-10m-sentence-corpus`
  - Pod: `/workspace/rust-pcn/data/smclm-10m-sentence-corpus`
  - Bridge dirs:
    - Host: `/home/kadajett/dev/rust-pcn/data/sentence-bridge-smclm`
    - Pod: `/workspace/rust-pcn/data/sentence-bridge-smclm`
  - Sharded bridge dirs (current v5):
    - Host: `/home/kadajett/dev/rust-pcn/data/sentence-bridge-smclm-sharded`
    - Pod: `/workspace/rust-pcn/data/sentence-bridge-smclm-sharded`
  - Snapshot stats: `~7,958,341` lines across 10 `.txt` shards

- Existing mixed books/code corpus
  - Host: `/home/kadajett/dev/rust-pcn/data/books`
  - Pod: `/workspace/rust-pcn/data/books`

- Reasoning Q/A corpus (HF `nohurry/Opus-4.6-Reasoning-3000x-filtered`)
  - Source schema: `problem`, `thinking`, `solution`
  - Policy: ignore `thinking`; keep only `problem` + `solution`
  - Recommended output:
    - Host: `/home/kadajett/dev/rust-pcn/data/reasoning-qa-sharded`
    - Pod: `/workspace/rust-pcn/data/reasoning-qa-sharded`
  - Prep command:
    - `python3 scripts/prepare-reasoning-corpus.py --out-dir /home/kadajett/dev/rust-pcn/data/reasoning-qa-sharded --shard-size 20000`
  - Format emitted per sample:
    - `Question:\n...\n\nAnswer:\n...`
  - Manifest:
    - `/home/kadajett/dev/rust-pcn/data/reasoning-qa-sharded/manifest.json`
  - Current prepared snapshot:
    - `lines_seen=2326`
    - `examples_written=2249`
    - `thinking_included=false`

## Post-1M Resume Handoff (Reasoning Phase)

- Use this after the current 1M v7 run completes:
  - `scripts/run-reasoning-resume.sh --checkpoint-dir /workspace/erm-rust/data/experiments/alice-run-b2-m1m-v7-sharded-3phase-r1/checkpoints --exp-id alice-run-b2-m1m-v7-reasoning-r1 --data /workspace/rust-pcn/data/reasoning-qa-sharded --add-steps 200000 --backend cuda --erm-bin /workspace/erm-rust/bin/erm.new`
- Script ensures:
  - resume from `checkpoints/latest`
  - no chain-of-thought field ingestion
  - same checkpoint continuity while changing data domain

## Canonical Pod Pull + Analysis Workflow

Use this exact sequence whenever diagnosing "plateau", stalls, or regressions.

1. Identify live pod and basic status.
   - `kubectl get pod -n pcn-train -l job-name=<job> -o wide`
   - `kubectl logs -n pcn-train <pod> --tail=120`
2. Verify writer activity and step progression.
   - `kubectl exec -n pcn-train <pod> -- wc -l <metrics.jsonl>`
   - `kubectl exec -n pcn-train <pod> -- tail -n 20 <metrics.jsonl>`
   - Repeat after ~10-20s. If line count is unchanged, check process state.
3. Check trainer process health.
   - `kubectl exec -n pcn-train <pod> -- ps -eo pid,pcpu,pmem,rss,etime,cmd | grep 'erm.new diffusion-train'`
   - `kubectl exec -n pcn-train <pod> -- top -b -n 1 | head -20`
4. Check GPU memory/utilization quickly.
   - `kubectl exec -n pcn-train <pod> -- nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader`
5. Pull metrics for cross-run comparisons (local temp).
   - `kubectl cp -n pcn-train <pod>:/workspace/erm-rust/data/experiments/<exp>/checkpoints/metrics.jsonl /tmp/erm-metrics/<exp>.metrics.jsonl`
6. Compute trend windows with `jq` + `awk`.
   - Must at least compute:
     - `mean(<=500)`
     - `loss@2000`
     - `mean(1500-2000)`
     - `slope(1500-2000)`
     - `mean(>2000)` and `slope(>2000)` when available
     - `deaths` mean/max in the same windows
7. Compare against last 2-3 experiments using same windows.
   - Never compare different windows directly.
8. Record result in `experiments/tracking/theory-tracker.json`.
   - Add qualitative stage note (for example "the-stage collapse") if present.
   - Keep `evaluation.status` as `running` until run is actually ended.

## Plateau Interpretation Rules (Use Consistently)

- Do not call hard plateau at ~2k alone.
- Treat as "noisy shelf" if:
  - `slope(1500-2000)` or `slope(>2000)` is near 0 but slightly negative
  - and fresh local minima continue to appear.
- Treat as likely plateau/regression if:
  - `slope(>2000)` turns positive over a long enough window (>= 1k steps),
  - and `last_2k_mean` does not improve by at least ~5%.
- Always check if throughput is stalled separately from loss plateau:
  - unchanged `metrics.jsonl` line count + unchanged step in logs.

## One-Command Readable Output Tail

- Command: `ermiotail` (alias to `ermio`) from `~/.zshrc`
- Behavior: auto-select newest running `erm-alice-run-*` pod and stream readable `clean/corr/pred` lines from latest checkpoint json/jsonl.
