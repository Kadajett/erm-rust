# ERM Multi-Experiment Training Suite

Deployed: 2026-03-02T16:31Z

## GPU Setup
- **Node**: theshire (1 physical GPU, time-sliced to 4 virtual slots via NVIDIA device plugin)
- **Existing workload**: pcn-sprite-train-half (1 GPU slot)
- **Available for experiments**: 3 GPU slots
- **Strategy**: 3 experiments run concurrently, 3 queued (auto-start as slots free)

## Experiments

| ID | seq_len | hidden_dim | num_heads | num_ants | emax | batch | Description |
|----|---------|-----------|-----------|----------|------|-------|-------------|
| exp-a | 512 | 192 | 6 | 64 | 8 | 2 | Baseline long-context |
| exp-b | 384 | 256 | 8 | 128 | 16 | 2 | Wide model, many ants |
| exp-c | 256 | 256 | 8 | 256 | 16 | 1 | Max ants, short context |
| exp-d | 512 | 160 | 5 | 96 | 12 | 2 | Lean long-context |
| exp-e | 384 | 192 | 6 | 64 | 16 | 4 | High batch throughput |
| exp-f | 256 | 128 | 4 | 128 | 8 | 4 | Small+fast throughput |

All experiments: 10,000 steps, log-every=50, checkpoint-every=500 (auto from erm binary).

## Outputs
Per experiment: `/home/kadajett/dev/erm-rust/data/experiments/<exp-id>/`
- `checkpoints/` — graph.json, ant_state.json, colony_meta.json, snapshots/
- `logs/colony-train.log` — full training log
- `config.json` — experiment config
- `warmstart/` — saved on clean exit

## Monitoring
```bash
python3 /home/node/.openclaw/workspace/erm-rust/experiments/status.py
```

## Manifests
- `erm-experiments.json` — K8s Job definitions
- `train-script.sh.tpl` — Training script template
- `gen.py` — Manifest generator

## Previous Training
- `erm-books-train-gpu` terminated at step ~8050 (checkpoint saved)
- Backup at: `/home/kadajett/dev/erm-rust/data/colony-checkpoints/pre-termination-backup/`
- Warmstart preserved at: `/home/kadajett/dev/erm-rust/data/colony-checkpoints/warmstart/`
