# erm-rust

**Emergent Route Model** — a non-autoregressive, non-diffusion text generator that learns via stigmergic exploration and exploitation.

## What is ERM?

ERM generates text through iterative parallel refinement. Instead of predicting tokens left-to-right (autoregressive) or gradually denoising a continuous distribution (diffusion), ERM:

1. **Starts from noise** — initializes a sequence of MASK tokens
2. **Refines in parallel** — each step, a neural scorer proposes edits to any position simultaneously
3. **Routes via pheromones** — a persistent sparse graph (the "route graph") encodes which token positions provide useful evidence for predicting other positions, updated stigmergically like ant colony optimization
4. **Converges** — stops when no further improvements are made or confidence exceeds a threshold

### The Two-Brain Architecture

ERM has two components that work together:

| Component | Type | Purpose |
|---|---|---|
| **Neural Scorer Sθ** | Static learned parameters | Given current tokens + route messages, produces per-position token distributions and uncertainty |
| **Route Graph G** | Dynamic sparse memory | Encodes pheromone trails between positions; shapes information flow without adding parameters |

The scorer looks like a small encoder-only transformer (~6 blocks, d=256). What makes ERM unusual is that the *effective computation graph* is shaped by G — which grows and evolves during training and inference. ERM scales by refining G, not by scaling θ.

### Ant Colony Mechanics

At each refinement step, two ant populations operate:

- **Leaders (10%)** — explore high-uncertainty, low-pheromone positions; propose new graph edges; take risks
- **Followers (90%)** — exploit high-pheromone routes; propose conservative edits along strong edges

Ants that fail to improve the objective for K consecutive actions "die," taint their traversed routes (suppressing those edges), and are replaced. This enforces continual exploration and prevents the model from getting stuck.

---

## How ERM Relates to PCN

This project lives alongside [rust-pcn](../rust-pcn), a Predictive Coding Network implementation in Rust/burn. Both are trained in parallel for direct comparison.

| Property | ERM (this repo) | PCN (rust-pcn) |
|---|---|---|
| **Paradigm** | Stigmergic refinement | Predictive coding / energy minimization |
| **Inference** | Iterative parallel denoising | Forward pass (after training) |
| **Memory** | Dynamic route graph G | Static learned weights |
| **Scalability** | Grow graph structure | Scale parameters |
| **Training signal** | Denoising loss + pheromone updates | Prediction error minimization |
| **Locality** | Route-conditioned attention | Layer-local error signals |

**Why compare them?** Both models reject the standard autoregressive paradigm but take fundamentally different approaches. PCN draws from neuroscience (Bayesian brain, hierarchical prediction). ERM draws from swarm intelligence (stigmergy, emergence). Training them on the same data and evaluating them on the same benchmarks should reveal how these different inductive biases affect generation quality, convergence, and sample efficiency.

---

## Hardware Target

- **GPU:** RTX 3050 (4GB VRAM)
- **Backend:** [burn](https://github.com/tracel-ai/burn) with CUDA
- **Language:** Rust

VRAM is the primary constraint. See `TODO.md` for memory budgets and mitigation strategies.

---

## Project Status

🚧 **Pre-implementation planning phase.** See `TODO.md` for the full milestone breakdown.

---

## Repository Structure (planned)

```
erm-rust/
├── README.md           # This file
├── TODO.md             # Full milestone plan
├── Cargo.toml          # Workspace manifest (TBD)
├── src/
│   ├── main.rs         # CLI entrypoint
│   ├── model/          # Neural scorer Sθ
│   ├── graph/          # Route graph G, CSR/dense storage
│   ├── ant/            # Leader/follower ant logic
│   ├── training/       # Loss, pheromone updates, data pipeline
│   └── inference/      # Generation loop
├── benches/            # Criterion benchmarks
├── tests/              # Integration tests
└── .github/
    └── workflows/      # CI
```

---

## Quick Links

- Architecture plan: `docs/architecture-plan.md` (source document)
- PCN comparison: see `TODO.md` → Phase 8
- Memory budget: see `TODO.md` → VRAM notes throughout
