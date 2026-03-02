# Improvements Branch: `improvements-automated`

## Changes Made

### 1. Renderer 2D Force-Directed Layout
- Replaced 1D horizontal line layout with deterministic Fruchterman-Reingold
- 50 iterations, nodes start on circle, settle via repulsion + attraction
- Canvas: 1200x800, Bezier edge curves with perpendicular offset
- Commit: `e628dff`

### 2. Seed RouteGraph Skip-Connections
- `RouteGraph::new()` now pre-seeds edges at offsets ±1, ±2, ±4
- Eliminates cold-start period where colony has no structure
- Added `new_empty()` constructor for tests needing explicit topology
- Commit: `9415721`

### 3. Colony Batch Loop
- Colony step now processes ALL batch items (was only batch_idx=0)
- Each batch item gets independent proposals, merges, and edits
- Graph receives more diverse learning signal per training step
- Commit: `79301cb`

### 4. Warmstart Checkpoint Save/Load
- `ColonyTrainer::save_warmstart()` — saves scorer weights (burn BinFileRecorder), graph, ant_state, config
- `ColonyTrainer::load_warmstart()` — restores full trainer state
- Automatically saved when `--checkpoint-dir` is provided
- Commit: `a5e3b6f`

### 5. Live I/O Example
- New `io-example` CLI subcommand
- Loads warmstart checkpoint, corrupts batch, runs scorer + colony
- Outputs JSON with clean/corrupted text, top-3 predictions, colony stats
- Commit: `af0bb1e`

## Artifacts (not in git, generated locally)

- **Warmstart checkpoint:** `data/colony-checkpoints/warmstart/`
  - `scorer.bin` (81KB) — burn scorer weights
  - `graph.json` — RouteGraph state
  - `ant_state.json` — ant colony state
  - `config.json` — config used for training
- **I/O Example:** `data/colony-checkpoints/io_example.json`
- **Snapshots:** `data/colony-checkpoints/snapshots/step_00100.json`, `step_00200.json`
- **Training data:** `data/train_small.txt` (72KB, repeated proverbs)

## Training Results (200 steps, CPU, small config)

- **Loss:** 3.80 → 2.98 (21% reduction)
- **Mean φ:** 0.021 (early training, pheromone building up)
- **Vocab:** 36 chars, seq_len=32, hidden_dim=32
- **Model correctly learns character frequency distribution** (space, 'e', 'o' as top predictions)

## Reproducing

```bash
# Train:
cargo run --release -- colony-train --data data/train_small.txt --steps 200 --config data/small_config.json --backend cpu --checkpoint-dir data/colony-checkpoints

# Generate I/O example:
cargo run --release -- io-example --data data/train_small.txt --checkpoint data/colony-checkpoints/warmstart --config data/small_config.json --output data/colony-checkpoints/io_example.json
```
