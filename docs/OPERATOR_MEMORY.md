# Operator Memory (Shared: Codex + Claude)

Last updated: 2026-03-05 UTC (03:28)

## Current Live Run

- Job: `erm-alice-run-m1m-v7-i07-r1-resume`
- Experiment id: `alice-run-b2-m1m-v7-sharded-3phase-r1-i07-r1`
- Status: running (resume canary on Burn CUDA from step `211250`)
- Confirmed phase/data order:
  - Phase 1: `100000` steps on `/workspace/rust-pcn/data/english-frontload-sharded`
  - Phase 2: `200000` steps on `/workspace/rust-pcn/data/sentence-bridge-smclm-sharded`
  - Phase 3: `700000` steps on `/workspace/rust-pcn/data/books`
- Planned post-1M extension order:
  - Reasoning Q/A resume corpus: `/workspace/rust-pcn/data/reasoning-qa-sharded`
  - Million numbers curriculum (3 variants): `/workspace/rust-pcn/data/million-english-numbers-sharded/...`

Observed early-phase behavior:
- Entered classic early "the-stage" collapse in predictions (high-frequency stopword loops).
- Around steps `~2k-2.5k`, loss hovered in `~7.0-7.9` band with noisy but slight net downtrend.

Ticket #6 validation notes (2026-03-04 UTC):
- Code commit on `main`: `6b6e3b2` (graph-health metrics in `metrics.jsonl` + AIM watcher tracked keys update).
- Live snapshots captured (no interruption):
  - step `158750`:
  - `/home/kadajett/.openclaw/workspace/erm-rust/data/checkpoint-snapshots/alice-run-b2-m1m-v7-sharded-3phase-r1-20260304T214242Z`
  - step `175250`:
  - `/home/kadajett/.openclaw/workspace/erm-rust/data/checkpoint-snapshots/alice-run-b2-m1m-v7-sharded-3phase-r1-20260304T231551Z`
- Resume smoke with new exp id attempted:
  - `alice-run-b2-m1m-v7-sharded-3phase-r1-i06-r1` (rolled back locally due CPU throughput on large runtime vocab).
- Resume redeploy attempts `i06-r2`..`i06-r5` failed due container/runtime setup issues (`python3` missing and then `libcuda.so` load failure).
- Current canary `i06-r6` is healthy:
  - CUDA preflight passes (`libcuda.so` visible, `nvidia-smi` works).
  - logs show `Backend: CUDA device 0` and `Resumed from checkpoint`.
  - `metrics.jsonl` is writing from step `175260+`.
- AIM sidecar deployment `aim-sidecar-live-v7` is retargeted to `alice-run-b2-m1m-v7-sharded-3phase-r1-i06-r6`.
- Fast schema smoke (local CPU) completed:
  - `ticket6-graph-health-smoke-r2`
  - verified new fields in `metrics.jsonl`: taint/age/clamp/entropy/top1/leader-survival/prune/insert.

Ticket #1 rollout notes (2026-03-04 UTC):
- Code commit on `main`: `78c7289` (`phi_min` + elite-only pheromone deposits).
- Source run before redeploy:
  - job `erm-alice-run-m1m-v7-i06-r6-resume`, latest checkpoint step `178750`.
- Snapshot captured before stop/rebuild:
  - `/home/kadajett/.openclaw/workspace/erm-rust/data/checkpoint-snapshots/alice-run-b2-m1m-v7-sharded-3phase-r1-i06-r6-20260304T234921Z`
- New canary deployment:
  - job `erm-alice-run-m1m-v7-i01-r1-resume`
  - exp `alice-run-b2-m1m-v7-sharded-3phase-r1-i01-r1`
  - resumed from `/workspace/erm-rust/data/experiments/alice-run-b2-m1m-v7-sharded-3phase-r1-i06-r6/checkpoints/latest` at step `179000`
- Additional safety snapshots captured during ticket #1 run:
  - `/home/kadajett/.openclaw/workspace/erm-rust/data/checkpoint-snapshots/alice-run-b2-m1m-v7-sharded-3phase-r1-i01-r1-20260305T005035Z`
  - `/home/kadajett/.openclaw/workspace/erm-rust/data/checkpoint-snapshots/alice-run-b2-m1m-v7-sharded-3phase-r1-i01-r1-20260305T005745Z`
- Ticket #1 config values injected in `train-config.json`:
  - `phi_min = 0.0001`
  - `elite_k = 10` (10% of 96 ants)
- AIM sidecar deployment `aim-sidecar-live-v7` is retargeted to `alice-run-b2-m1m-v7-sharded-3phase-r1-i01-r1`.

Ticket #2 rollout notes (2026-03-05 UTC):
- Code commit on `main`: `e3760fb` (route aggregation utility weighting).
- Source run before redeploy:
  - job `erm-alice-run-m1m-v7-i01-r1-resume`, latest checkpoint step `189750`.
- New canary deployment:
  - job `erm-alice-run-m1m-v7-i02-r1-resume`
  - exp `alice-run-b2-m1m-v7-sharded-3phase-r1-i02-r1`
  - resumed from `/workspace/erm-rust/data/experiments/alice-run-b2-m1m-v7-sharded-3phase-r1-i01-r1/checkpoints/latest` at step `190000`
- Ticket #2 config values injected in `train-config.json`:
  - `route_kappa_utility = 0.5`
  - `elite_k = 10`
  - `phi_min = 0.0001`
- AIM sidecar deployment `aim-sidecar-live-v7` is retargeted to `alice-run-b2-m1m-v7-sharded-3phase-r1-i02-r1`.

Ticket #3 rollout notes (2026-03-05 UTC):
- Code commit on `main`: `ef9b8cc` (ACS local pheromone update control).
- Source run before redeploy:
  - job `erm-alice-run-m1m-v7-i02-r1-resume`, latest checkpoint step `194750`.
- Snapshot captured before stop/rebuild:
  - `/home/kadajett/.openclaw/workspace/erm-rust/data/checkpoint-snapshots/alice-run-b2-m1m-v7-sharded-3phase-r1-i02-r1-20260305T013212Z`
- New canary deployment:
  - job `erm-alice-run-m1m-v7-i03-r1-resume`
  - exp `alice-run-b2-m1m-v7-sharded-3phase-r1-i03-r1`
  - resumed from `/workspace/erm-rust/data/experiments/alice-run-b2-m1m-v7-sharded-3phase-r1-i02-r1/checkpoints/latest` at step `195000`
- Ticket #3 config values injected in `train-config.json`:
  - `local_update_xi = 0.01`
  - `route_kappa_utility = 0.5`
  - `elite_k = 10`
  - `phi_min = 0.0001`
- AIM sidecar deployment `aim-sidecar-live-v7` is retargeted to `alice-run-b2-m1m-v7-sharded-3phase-r1-i03-r1`.

Ticket #4 rollout notes (2026-03-05 UTC):
- Code commit on `main`: `5d9fafb` (step-dependent pheromone schedule hooks).
- Source run before redeploy:
  - job `erm-alice-run-m1m-v7-i03-r1-resume`, latest checkpoint step `195750`.
- Snapshot captured before stop/rebuild:
  - `/home/kadajett/.openclaw/workspace/erm-rust/data/checkpoint-snapshots/alice-run-b2-m1m-v7-sharded-3phase-r1-i03-r1-20260305T014227Z`
- New canary deployment:
  - job `erm-alice-run-m1m-v7-i04-r1-resume`
  - exp `alice-run-b2-m1m-v7-sharded-3phase-r1-i04-r1`
  - resumed from `/workspace/erm-rust/data/experiments/alice-run-b2-m1m-v7-sharded-3phase-r1-i03-r1/checkpoints/latest` at step `196000`
- Ticket #4 config values injected in `train-config.json`:
  - `pheromone_schedule_mode = linear`
  - `schedule_evap_mult_start = 1.5`, `schedule_evap_mult_end = 0.7`
  - `schedule_route_lambda_mult_start = 0.7`, `schedule_route_lambda_mult_end = 1.3`
  - `schedule_diversity_penalty_mult_start = 0.8`, `schedule_diversity_penalty_mult_end = 1.0`
- AIM sidecar deployment `aim-sidecar-live-v7` is retargeted to `alice-run-b2-m1m-v7-sharded-3phase-r1-i04-r1`.
- Final i04 canary window (step `196010 -> 206750`) stayed moderately exploratory (entropy mean `~0.85`, top-1 share mean `~0.62`) but leader edges remained `0`.

Ticket #5 rollout notes (2026-03-05 UTC):
- Code commit on `main`: `ee1f3f3` (age-based eta schedule with half-life option).
- Source run before redeploy:
  - job `erm-alice-run-m1m-v7-i04-r1-resume`, latest checkpoint step `206750`.
- Snapshot captured before stop/rebuild:
  - `/home/kadajett/.openclaw/workspace/erm-rust/data/checkpoint-snapshots/alice-run-b2-m1m-v7-sharded-3phase-r1-i04-r1-20260305T025121Z`
- New canary deployment:
  - job `erm-alice-run-m1m-v7-i05-r1-resume`
  - exp `alice-run-b2-m1m-v7-sharded-3phase-r1-i05-r1`
  - resumed from `/workspace/erm-rust/data/experiments/alice-run-b2-m1m-v7-sharded-3phase-r1-i04-r1/checkpoints/latest` at step `206750`
- Ticket #5 config values injected in `train-config.json`:
  - `age_eta_schedule = half_life`
  - `age_half_life = 128.0`
  - retained ticket #4 schedule controls (`linear`, evap/lambda/diversity multipliers)
- AIM sidecar deployment `aim-sidecar-live-v7` is retargeted to `alice-run-b2-m1m-v7-sharded-3phase-r1-i05-r1`.
- Final i05 audit window (`206760 -> 211160`) remained stable but still showed `leader_edges = 0` (entropy `~0.85`, top-1 share `~0.62`).

Leader-edge wiring fix rollout (2026-03-05 UTC):
- Code commit on `main`: `d3402ba` (diffusion loop now inserts leader edges via `propose_edges` and updates leader utility EMA each step).
- Source run before redeploy:
  - job `erm-alice-run-m1m-v7-i05-r1-resume`, latest checkpoint step `211000`.
- Snapshot captured before stop/rebuild:
  - `/home/kadajett/.openclaw/workspace/erm-rust/data/checkpoint-snapshots/alice-run-b2-m1m-v7-sharded-3phase-r1-i05-r1-20260305T032350Z`
- New canary deployment:
  - job `erm-alice-run-m1m-v7-i07-r1-resume`
  - exp `alice-run-b2-m1m-v7-sharded-3phase-r1-i07-r1`
  - resumed from `/workspace/erm-rust/data/experiments/alice-run-b2-m1m-v7-sharded-3phase-r1-i05-r1/checkpoints/latest` at step `211250`
- AIM sidecar deployment `aim-sidecar-live-v7` is retargeted to `alice-run-b2-m1m-v7-sharded-3phase-r1-i07-r1`.
- Initial i07 audit window (`211260 -> 211390`) confirms leader-edge fix is working: `leader_edges` now non-zero (mean `~522`), leader-edge fraction `~0.67`, survival `~0.71`.

### CUDA/Burn Setup (Known-Good)

- Use `--backend cuda` for `erm.new diffusion-train` (this is the Burn CUDA path).
- On this cluster, CUDA resume pods must include:
  - `runtimeClassName: nvidia`
  - `resources.requests/limits.nvidia.com/gpu: "1"`
  - `LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu`
  - CUDA hostPath mounts for `/usr/local/cuda-13.0` and `/usr/local/cuda-13.1` (mounted at `/usr/local/cuda`)
- Do not mount host `/usr/lib/x86_64-linux-gnu` over the container filesystem for training pods; it can break apt/package setup.

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

- Million numbers corpus (HF `lsb/million-english-numbers`)
  - Source schema: `text` only (word form of integer sequence from `0` to `999999`)
  - Prepared output root:
    - Host: `/home/kadajett/dev/rust-pcn/data/million-english-numbers-sharded`
    - Pod: `/workspace/rust-pcn/data/million-english-numbers-sharded`
  - Variants:
    - `word-copy-sharded` (denoise-style reconstruction)
      - Host: `/home/kadajett/dev/rust-pcn/data/million-english-numbers-sharded/word-copy-sharded`
      - Pod: `/workspace/rust-pcn/data/million-english-numbers-sharded/word-copy-sharded`
      - Template: `<number-word>`
    - `int-to-word-sharded`
      - Host: `/home/kadajett/dev/rust-pcn/data/million-english-numbers-sharded/int-to-word-sharded`
      - Pod: `/workspace/rust-pcn/data/million-english-numbers-sharded/int-to-word-sharded`
      - Template: `Input:\n<integer>\n\nOutput:\n<number-word>`
    - `word-to-int-sharded`
      - Host: `/home/kadajett/dev/rust-pcn/data/million-english-numbers-sharded/word-to-int-sharded`
      - Pod: `/workspace/rust-pcn/data/million-english-numbers-sharded/word-to-int-sharded`
      - Template: `Input:\n<number-word>\n\nOutput:\n<integer>`
  - Prep command:
    - `python3 scripts/prepare-million-english-numbers.py --out-root /home/kadajett/dev/rust-pcn/data/million-english-numbers-sharded --shard-size 20000 --progress-every 200000`
  - Verified snapshot:
    - `rows_seen=1,000,000`
    - `examples_written=1,000,000` per variant
    - `shard_count=50` per variant
    - Boundary checks passed at indices `0`, `19999`, `20000`, `999999`

## Post-1M Resume Handoff (Reasoning Phase)

- Use this after the current 1M v7 run completes:
  - `scripts/run-reasoning-resume.sh --checkpoint-dir /workspace/erm-rust/data/experiments/alice-run-b2-m1m-v7-sharded-3phase-r1/checkpoints --exp-id alice-run-b2-m1m-v7-reasoning-r1 --data /workspace/rust-pcn/data/reasoning-qa-sharded --add-steps 200000 --backend cuda --erm-bin /workspace/erm-rust/bin/erm.new`
- Script ensures:
  - resume from `checkpoints/latest`
  - no chain-of-thought field ingestion
  - same checkpoint continuity while changing data domain

## Save + Backup + Resume-With-New-Data Playbook

Use this when we want to preserve current training state, then continue learning on a new dataset.

### 1) Snapshot current training safely (no interruption)

1. Capture live pod and experiment variables:
   - `job=erm-alice-run-m1m-v7-sharded-3phase`
   - `pod=$(kubectl get pods -n pcn-train -l job-name=${job} -o jsonpath='{.items[0].metadata.name}')`
   - `exp=alice-run-b2-m1m-v7-sharded-3phase-r1`
2. Read latest checkpointed step:
   - `kubectl exec -n pcn-train ${pod} -- cat /workspace/erm-rust/data/experiments/${exp}/checkpoints/latest/step.json`
3. Copy a timestamped backup snapshot locally:
   - `ts=$(date -u +%Y%m%dT%H%M%SZ)`
   - `out=/home/kadajett/.openclaw/workspace/erm-rust/data/checkpoint-snapshots/${exp}-${ts}`
   - `mkdir -p "${out}"`
   - `kubectl cp -n pcn-train "${pod}:/workspace/erm-rust/data/experiments/${exp}/checkpoints/latest" "${out}/latest"`
   - `kubectl cp -n pcn-train "${pod}:/workspace/erm-rust/data/experiments/${exp}/checkpoints/metrics.jsonl" "${out}/metrics.jsonl"`
   - `kubectl cp -n pcn-train "${pod}:/workspace/erm-rust/data/experiments/${exp}/bpe_vocab.json" "${out}/bpe_vocab.json"`
4. Verify snapshot:
   - `cat "${out}/latest/step.json"`
   - `test -f "${out}/latest/model.safetensors" && echo "model ok"`
   - `wc -l "${out}/metrics.jsonl"`

### 2) Optional: wait for a fresh checkpoint before stopping/redeploying

- Checkpoint cadence is currently `250` steps. If we need a cleaner cutoff:
  - watch logs for `checkpoint saved` near desired step before stopping:
  - `kubectl logs -n pcn-train ${pod} --tail=200 | grep "checkpoint saved"`

### 3) Resume on new data (same weights, new data domain)

- Preferred generic resume entrypoint:
  - `scripts/run-diffusion-milestone.sh --checkpoint-dir /workspace/erm-rust/data/experiments/${exp}/checkpoints --data <new_data_dir> --exp-id <new_exp_id_or_same> --add-steps <N> --backend cuda --erm-bin /workspace/erm-rust/bin/erm.new`
- Reasoning-specific shortcut:
  - `scripts/run-reasoning-resume.sh --checkpoint-dir /workspace/erm-rust/data/experiments/${exp}/checkpoints --exp-id <new_exp_id_or_same> --data /workspace/rust-pcn/data/reasoning-qa-sharded --add-steps 200000 --backend cuda --erm-bin /workspace/erm-rust/bin/erm.new`

Notes:
- If user asks for a **new experiment id** but still says **resume**, keep `--resume` and write to a new experiment path for clean AIM separation.
- If user asks for a **new experiment id** and a **full restart**, do not pass `--resume`.

### 4) Auto data loading (new `.txt` files during active training)

- Status: still enabled and tested.
  - Code: `erm-train/src/streaming_dataset.rs`
  - Rescan interval: `NEW_FILE_RESCAN_INTERVAL = 8` processed files.
  - Discovery log line:
    - `[streaming_dataset] discovered <N> new .txt files in <data_dir>`
- Validation test:
  - `cargo test -p erm-train test_discover_new_txt_paths -- --nocapture`
  - Last run status: `pass` on `2026-03-04`.

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

## Offline Playground (No Training Disturbance)

- Goal: play with current weights while active training continues.
- Rule: use CPU backend for playground (`--backend cpu`) so GPU training run is unaffected.

### 1) Snapshot weights from running pod

- Example snapshot flow:
  - `pod=$(kubectl get pods -n pcn-train -l job-name=erm-alice-run-m1m-v7-sharded-3phase -o jsonpath='{.items[0].metadata.name}')`
  - `exp=alice-run-b2-m1m-v7-sharded-3phase-r1`
  - `ts=$(date -u +%Y%m%dT%H%M%SZ)`
  - `out=/home/kadajett/.openclaw/workspace/erm-rust/data/checkpoint-snapshots/${exp}-${ts}`
  - `kubectl cp -n pcn-train "${pod}:/workspace/erm-rust/data/experiments/${exp}/checkpoints/latest" "${out}/latest"`
  - `kubectl cp -n pcn-train "${pod}:/workspace/erm-rust/data/experiments/${exp}/bpe_vocab.json" "${out}/bpe_vocab.json"`

### 2) Run perpetual 1Hz diffusion loop

- New CLI mode: `infer-live` (stop with Ctrl+C).
- Example:
  - `target/debug/erm infer-live --checkpoint <snapshot>/latest --length 96 --steps 2 --backend cpu --interval-ms 1000 --feedback-tokens 24 --prompt "demo prompt"`
- Behavior:
  - prints one diffusion output per second forever
  - feeds back leading tokens from previous output for iterative drift
