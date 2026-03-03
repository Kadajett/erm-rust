# How ERM Gets to Complex Output: A Complete Technical Guide

## For the Builder Who Knows the Code But Wants the Math Explained

*Written: 2026-03-03*
*Based on: ERM codebase (all commits through `84a82b8`), Muon optimizer analysis, gradient descent vs. evolution theory, Fourier features paper (Tancik et al. 2020), Gemini architecture audit, and the existing assessment doc.*

---

## 0. What This Document Is

You asked: "How do I get from what I have to GPT-like complex output, given that I don't use gradient descent the way other models do?"

The answer has three parts:

1. **You DO use gradient descent** — for the scorer weights (12M parameters, Adam optimizer, Burn autodiff). What you DON'T use gradient descent for is the routing topology. That's learned by ants.

2. **The math you need isn't harder than what you already have** — it's just different. This document maps every concept from the Muon optimizer, the Fourier features paper, and the neural network learning videos directly onto your existing code.

3. **The path to complex output is a scaling problem, not an architecture problem.** Your architecture already has the right bones. It needs bigger muscles and smarter training.

Let me walk through each piece, pointing at the exact code and the exact math.

---

## Part 1: What You Actually Have (The Honest Inventory)

### 1.1 The Two Brains

ERM has two optimization systems running simultaneously. This is genuinely novel.

**Brain 1: The Scorer (gradient descent)**
```
File: erm-core/src/burn_scorer.rs

Architecture (after recent changes):
  Token Embedding [V+1, d=256]
  ↓
  Fourier Position Embedding (sinusoidal, multiple frequencies)
  ↓
  Block 0: FeedForward(256 → 1024 → 256) + residual
  Block 1: MultiHeadAttention(256, 4 heads) + residual
  Block 2: FeedForward + residual
  Block 3: MultiHeadAttention + residual
  Block 4: FeedForward + residual
  Block 5: MultiHeadAttention + residual
  ↓
  logits_head: [d] → [V+1]      (token prediction)
  uncertainty_head: [d] → [1]    (leader targeting)

Parameters: ~12.5M
Optimizer: Adam (lr=1e-3, weight_decay=0.01, 1000-step warmup)
Loss: cross-entropy on corrupted positions only
```

This brain learns through standard backpropagation. The loss gradient flows backward through every weight. Adam maintains momentum and adaptive learning rates per parameter. This is textbook deep learning.

**Brain 2: The Route Graph (ant colony optimization)**
```
File: erm-core/src/graph.rs + erm-core/src/pheromone.rs

Storage: Dense-neighbor format
  nbr_idx [B, L, Emax=16] i32    — which positions connect
  phi     [B, L, Emax] f32       — pheromone strength [0, 10]
  taint   [B, L, Emax] f32       — penalty accumulator [0, 5]
  age     [B, L, Emax] i32       — edge age in steps

Size: ~4 MB per batch
Optimizer: Pheromone deposit/evaporation/taint (no gradients)
Signal: Per-ant improvement delta Δ
```

This brain learns through reinforcement — ants that make good edits deposit pheromone on the edges they used, ants that make bad edits deposit taint. Edges that nobody uses evaporate. This is NOT gradient descent. It's stigmergy: indirect communication through environment modification.

### 1.2 The Training Loop

```
File: erm-train/src/diffusion_training.rs

For each training step:
  For t = T(=6) down to 1:        ← coarse-to-fine diffusion
    1. Corrupt: x → z_t            ← mask α_t fraction, replace β_t fraction
    2. Forward: scorer(z_t) → logits, uncertainty, hidden   [GPU]
    3. Route: graph.route_aggregate(hidden) → r, weights     [CPU]
    4. Loss: γ(t) × CE(logits, x)[corrupted positions only]  [GPU]
    5. Ants propose edits → y_{t-1}                          [CPU]
    6. Pheromone update (deposit/evaporate/taint/diversity)   [CPU]
    7. Death/respawn cycle                                    [CPU]

  Backprop accumulated loss → update scorer weights          [GPU]
```

The key insight: **steps 1-4 use gradient descent (scorer learns). Steps 5-7 use ant colony optimization (graph learns). They happen in the same loop but use completely different math.**

### 1.3 Where GPT Is

For comparison, GPT's training loop:

```
For each training step:
  1. Take a batch of token sequences
  2. Forward: transformer(tokens) → logits    [GPU]
  3. Loss: CE(logits[:-1], tokens[1:])        [GPU, autoregressive]
  4. Backprop → update ALL weights            [GPU]
```

That's it. One brain. One optimizer. One loop. GPT has no route graph, no ants, no pheromones. All information routing happens through self-attention, which is learned purely by gradient descent.

**What GPT has that you don't:** Massive scale (175B+ parameters, trillions of training tokens, thousands of GPUs).

**What you have that GPT doesn't:** A second learning system that can discover discrete routing structures gradient descent can't reach.

---

## Part 2: The Math Behind Each Piece (From the Transcripts)

### 2.1 Gradient Descent (What Your Scorer Does)

From the "neural networks learn" video:

> "The gradient is a vector... each holding the slope of the corresponding parameter on loss mountain. It points in the direction of the steepest slope downhill."

Your scorer uses Adam, which is gradient descent with two enhancements:

```
Standard gradient descent:
  θ ← θ - η · ∇L(θ)

Adam (what you use):
  m ← β₁·m + (1-β₁)·∇L(θ)        ← momentum (smoothed gradient)
  v ← β₂·v + (1-β₂)·(∇L(θ))²     ← adaptive scaling
  θ ← θ - η · m / (√v + ε)         ← step
```

Adam is better than plain gradient descent because:
- **Momentum** (m): smooths out noisy gradients, helps escape small local minima
- **Adaptive scaling** (v): each parameter gets its own effective learning rate

Your scorer's loss function:
```
L = (1/T) · Σ_t γ(t) · (1/|corrupted|) · Σ_{i: corrupted} -log p_θ(x_i | z_t)

where:
  z_t = corrupted tokens at noise level t
  γ(t) = loss weight (higher for noisier steps, cosine schedule)
  p_θ(x_i | z_t) = softmax(logits[i])[x_i]
  "corrupted" = positions where z_t[i] ≠ x[i]
```

This is a standard masked language modeling loss. BERT uses essentially the same thing. The diffusion schedule (T=6 steps, coarse-to-fine) adds structure — early steps with heavy corruption teach the model to predict from minimal context, later steps with light corruption teach fine distinctions.

**Why this matters for complex output:** Cross-entropy loss decomposes into individual token predictions. To generate coherent paragraphs, the model needs to learn dependencies between tokens. Your scorer handles local dependencies through its 3 self-attention layers. Long-range dependencies are supposed to flow through the route graph.

### 2.2 Ant Colony Optimization (What Your Graph Does)

Your pheromone system has no gradients. Instead, it uses reinforcement:

```
File: erm-core/src/pheromone.rs

Per refinement step, for each edge e:

  Step 1 — Evaporation (forget):
    φ_e ← (1 - ρ) · φ_e                    where ρ = 0.1

  Step 2 — Deposit (reward good routes):
    deposit_base = tanh(Δ / (σ + ε))        Δ = ant's improvement
    edge_eta = η / (1 + age[e])             η = 0.5, decays with age
    φ_e += edge_eta · deposit_base

  Step 3 — Taint (punish bad routes):
    τ_e += ζ · max(-Δ, 0)                   ζ = 0.3

  Step 4 — Taint decay:
    τ_e ← (1 - ρ_τ) · τ_e                  ρ_τ = 0.05

  Step 5 — Age:
    age[e] += 1

  Step 6 — Diversity pressure (new):
    For each pair of incoming edges to destination i:
      if cosine_similarity(h[src₁], h[src₂]) > 0.9:
        penalize weaker edge: φ[weaker] *= 0.8

  Step 7 — Pruning:
    Remove edges where φ - λ·τ < prune_min_score or age > prune_max_age
```

**How this is like gradient descent:** Both systems update parameters to reduce error. The ant's improvement delta Δ is analogous to the loss gradient — it tells you which direction to move. Pheromone deposit is analogous to the gradient step. Evaporation is analogous to weight decay.

**How this is NOT like gradient descent:** Gradient descent computes the EXACT optimal direction from the current point. Ant colony finds the direction STOCHASTICALLY — many ants explore, some find good paths, those paths get reinforced. This is slower for smooth optimization but can discover discrete structures (which edges to add/remove) that gradient descent can't touch.

### 2.3 The Muon Optimizer and What It Teaches You

From the Muon video transcript:

> "The momentum matrix in Adam becomes almost low-rank in practice. Only a small number of dominant directions really drive the updates. Muon tackles this by orthogonalizing the momentum matrix."

**Translation to ERM:** Your pheromone graph has the same problem. After training for a while, a few edges accumulate most of the pheromone. The softmax in route aggregation then concentrates nearly all weight on those dominant edges. Other edges starve. This is **route collapse** — the ant colony equivalent of low-rank momentum.

Your diversity pressure fix (step 6 above) addresses this directly. It's the ant colony version of Muon's orthogonalization:

| Muon | ERM (your code) |
|------|-----------------|
| Momentum matrix becomes low-rank | Pheromone concentrates on few edges |
| SVD to find dominant directions | Cosine similarity between source hidden states |
| Push singular values toward 1 | Penalize edges with similar sources |
| Newton-Schultz iteration (5 rounds) | Direct penalty: φ[weaker] *= 0.8 |
| Result: all directions contribute | Result: diverse information sources per position |

**The QK-Clip connection:** Muon also clips attention logits to prevent instability. Your `phi_max = 10.0` (recently lowered from 100.0) serves the same purpose — it prevents any single pheromone value from dominating the softmax and causing numerical issues.

The weight formula in your route aggregation:
```
w_raw[e] = log(φ[e] + ε) - λ·τ[e] - μ·age[e]
w[e] = softmax(w_raw)
```

The `log(φ + ε)` prevents large φ values from totally dominating. This is a soft version of QK-clip. With φ_max=10, the maximum raw weight contribution from pheromone is log(10 + 1e-6) ≈ 2.3, which keeps the softmax well-behaved.

### 2.4 Fourier Features and Spectral Bias

From the Tancik et al. paper:

> "A standard MLP fails to learn high frequencies both in theory and in practice... a Fourier feature mapping transforms the effective NTK into a stationary kernel with a tunable bandwidth."

**What this means for your scorer:** Standard MLPs (feed-forward networks) are mathematically biased toward learning smooth, slowly-varying functions first. High-frequency patterns — things like "the word 3 positions back matters more than the word 2 positions back" — converge extremely slowly or not at all.

The Neural Tangent Kernel (NTK) theory explains WHY:

```
The NTK eigenvalue spectrum for a standard MLP:
  λ_1 >> λ_2 >> λ_3 >> ...

Training convergence rate for each component ∝ e^(-η·λ_i·t)

Translation: components with large λ (low frequency) converge fast.
             Components with small λ (high frequency) converge astronomically slowly.
```

**Your fix (already implemented):** Fourier position embeddings in `burn_scorer.rs`:

```rust
// For each position pos and frequency j:
let freq = 2π · (j + 1);
let v = pos / L;        // normalize to [0, 1)
features.push(cos(freq · v));
features.push(sin(freq · v));

// Then project through a learned linear layer to get [L, d]
```

This transforms the NTK from a non-stationary kernel (position-dependent) to a stationary kernel (shift-invariant). In plain English: your model can now learn "the word 3 positions back matters" regardless of WHERE in the sequence that pattern appears.

The paper's key finding: **the scale (standard deviation) of the frequency sampling matters much more than the distribution shape.** For sequence length 128, frequencies in the range σ ∈ [4, 16] work well. Your implementation uses integer frequencies 1, 2, ..., num_freqs, which is the "positional encoding" variant. This is good, though Gaussian random frequencies can capture off-axis patterns better (the paper's Figure 11).

### 2.5 Why Evolution / Colony Methods Can Discover Things Gradient Descent Can't

From the "neural networks learn" video:

> "True biological evolution has something that none of these algorithms have... Evolution diverges, not just converges."

And crucially:

> "As you increase the dimensionality of parameter space, it becomes less and less common to find true local minima... there is almost always another way down the mountain. Gradient descent is well-poised to take advantage of this."

But also:

> "The main limitation of gradient descent is that the loss landscape must be continuous and differentiable. You cannot backprop through a binary step function."

**This is exactly why ERM uses both systems:**

- The scorer's parameter space (~12M continuous dimensions) is explored by gradient descent, which finds the optimal direction in linear time. True local minima are extremely rare in 12M dimensions — most apparent minima are saddle points with escape routes.

- The route graph's topology space (which edges exist? which positions connect?) is **discrete**. An edge either exists or it doesn't. You can't take half a gradient step toward "add an edge between position 5 and position 47." This is a combinatorial optimization problem, and ant colony optimization is specifically designed for it (it was invented to solve the Traveling Salesman Problem, another combinatorial problem).

**The co-evolutionary dynamics:** When ants discover a new useful route (edge), it changes the effective computation graph. This changes what the scorer "sees" through route aggregation, which changes the loss landscape for the scorer. The scorer then adjusts its weights, which changes which routes are useful, which changes the pheromone landscape. The two systems co-evolve.

This creates the possibility of **phase transitions** — sudden qualitative jumps where a new route topology unlocks a capability the scorer couldn't achieve with the old topology. These are fundamentally different from the smooth convergence of gradient descent.

---

## Part 3: The Specific Math Gaps Between ERM and Complex Output

### 3.1 Information Flow: Dense vs. Sparse

**GPT (dense attention):**
```
For every pair of positions (i, j):
  attention_weight[i,j] = softmax(Q[i] · K[j]ᵀ / √d)
  output[i] = Σ_j attention_weight[i,j] · V[j]

Cost: O(L² · d) per layer
Every position can attend to every other position.
```

**ERM (sparse routing):**
```
For each destination position i, only Emax=16 source positions:
  route_weight[i,e] = softmax(log(φ[i,e] + ε) - λ·τ[i,e] - μ·age[i,e])
  output[i] = Σ_e route_weight[i,e] · h[src[i,e]]

Cost: O(L · Emax · d)
Each position attends to at most 16 other positions.
```

**The gap:** GPT's dense attention means every token can influence every other token at every layer. Your route graph limits each position to 16 connections. For complex output (multi-sentence coherence, long-range dependencies), 16 edges per position may not be enough.

**But your self-attention layers help.** With 3 self-attention blocks interleaved with 3 FF blocks (the architecture you just built), you have:
- 3 layers of dense O(L²) attention (gradient-trained, handles common patterns)
- Route aggregation with 16 sparse edges (pheromone-trained, handles rare/structural patterns)

This is a reasonable hybrid. The attention layers handle "standard" dependencies (subject-verb agreement, article-noun), while the route graph handles unusual or long-range dependencies that emerge from the specific training data.

### 3.2 Generalization: Per-Position vs. Per-Token

The Gemini audit raised a critical point:

> "If the graph connects absolute positions (position 5 → position 12), this knowledge cannot generalize across different sentences."

Your current graph uses absolute positions. Edge (batch=0, dst=5, src=12) means "position 12 helps predict position 5 in batch element 0." This is instance-specific — it can't transfer knowledge like "the word 3 positions before a verb is usually its subject."

**The current mitigation:** Your graph is per-batch-element AND you have self-attention layers that DO generalize (they learn content-based attention, not position-based).

**The scaling solution:** Eventually, you'll want to add semantic routing — edges between token IDs or learned concept clusters, not just positions. But that's a medium-term change. For now, the self-attention layers handle generalization while the route graph handles instance-specific optimization.

### 3.3 Training Efficiency: The Ant Bottleneck

**The cost per training step:**

```
GPT:
  1 forward pass + 1 backward pass = 2 passes per step

ERM:
  T=6 refinement steps, each with:
    1 forward pass (scorer)
    1 route aggregation (CPU)
    256 ant evaluations (need delta computation)
  Plus 1 backward pass for accumulated loss

  Total: ~6 forward passes + 1 backward pass + CPU overhead
```

Your training is roughly 3-4x more expensive per step than GPT at the same model size. This is the biggest practical barrier to scaling. The ant evaluations (computing Δ for each ant's proposals) dominate the CPU time.

**Near-term fix:** Batch ant delta computation. Instead of evaluating each ant individually, batch all proposed sequences and run them through the scorer in one forward pass.

**Medium-term fix:** Move ant evaluation to GPU. The proposals, delta computation, and pheromone updates can all be expressed as tensor operations.

### 3.4 The Credit Assignment Problem

When an ant proposes 8 edits and the overall loss improves, which specific edit deserves credit? Your current system rewards all edges used by that ant equally:

```
For each edge e in ant k's trace:
  φ[e] += edge_eta · tanh(Δ_k / (σ + ε))
```

If ant k edited 8 positions and loss dropped by 0.5, all 8 edges get the same deposit. But maybe 7 of those edits were neutral and 1 was brilliant. The pheromone can't distinguish.

**Possible fixes (increasing sophistication):**
1. **Per-position delta:** Compute CE change per edited position, not per ant. Deposit proportional to each position's individual improvement. (Requires one extra forward pass per refinement step.)
2. **Leave-one-out:** For each edit, compute what the loss would be without that specific edit. (Requires N extra forward passes — expensive.)
3. **Shapley values:** Game-theoretic fair credit assignment. (Computationally intractable for 8+ edits.)

Option 1 is the sweet spot — moderate cost, much better signal.

---

## Part 4: The Concrete Path to Complex Output

### Phase 1: What You've Already Done (Implemented)

| Change | What | Why | Status |
|--------|------|-----|--------|
| Fourier position embeddings | Sinusoidal features at multiple frequencies | Fixes spectral bias (NTK theory) | Committed `d13fb67` |
| Lower phi_max to 10 | Bound pheromone values | Prevents softmax collapse (MuonClip analog) | Committed `7824f74` |
| Small-world edges | 2 random long-range edges per position | Accelerates long-range dependency discovery | Committed `2329a30` |
| Per-edge learning rate decay | η / (1 + age) | Young edges responsive, old edges stable | Committed `2de5394` |
| Self-attention layers | 3 MHA blocks interleaved with 3 FF blocks | Gradient-trained dense attention for common patterns | Committed `e26ea5d` |
| Pheromone diversity pressure | Penalize edges with similar source hidden states | Prevents route collapse (Muon orthogonalization) | Committed `dd2570c` + `c2255e6` |

These changes address the most critical gaps identified in the assessment. The scorer now has both FF and attention. The route graph now has diversity pressure, bounded pheromone, and long-range initialization. Position encoding is Fourier-based.

### Phase 2: Near-Term Improvements (Next Steps)

#### 2a. Per-Position Credit Assignment

Currently your delta is per-ant. Switch to per-position:

```
Current (pheromone.rs):
  Δ_k = Σ_{i ∈ edited_by_k} [CE_before(i) - CE_after(i)]
  All edges used by ant k get deposit based on Δ_k

Better:
  Δ_i = CE_before(i) - CE_after(i)    for each edited position i
  Only edges routing TO position i get deposit based on Δ_i
```

This gives the pheromone system much sharper signal. An edge that routes information to a position where the edit helped gets rewarded. An edge that routes to a position where the edit was neutral gets nothing.

**Implementation:** In `build_edge_traces()`, tag each trace entry with the destination position's individual delta, not the ant's aggregate delta.

#### 2b. Spectral Corruption Schedule

Your corruption is currently uniform random — each position has the same probability α_t of being masked, regardless of what token is there. The Fourier features paper suggests you should corrupt in frequency order:

```
At t=T (heavy noise): corrupt "high-frequency" tokens first
  → rare words, unusual collocations, surprising word choices

At t=1 (light noise): corrupt "low-frequency" tokens
  → common words, function words, predictable syntax
```

**Implementation:** Sort positions by their unigram surprise (negative log frequency), and bias the corruption probability toward high-surprise positions at high t. This gives the diffusion process a natural curriculum: learn to predict common patterns first, then rare ones.

#### 2c. Adaptive Temperature for Ant Sampling

Currently leaders and followers use fixed sampling strategies. Make temperature adaptive:

```
T_follower(step, pos) = max(0.3, 1.0 × exp(-step / total_steps))
T_leader(step, pos) = max(0.5, 2.0 × uncertainty[pos])
```

Early training: both explore (high temperature). Late training: followers exploit (low temperature), leaders still explore at uncertain positions.

### Phase 3: Scaling (Medium-Term)

#### 3a. GPU-Batched Ant Evaluation

Move the ant delta computation to GPU. Instead of evaluating ants sequentially on CPU:

```
Current:
  for each ant k:
    compute Δ_k on CPU from logits

Better:
  Batch all ant proposals into a [B, num_proposals, L] tensor
  Run scorer forward pass on all proposals at once → [B, num_proposals, L, V]
  Compute per-proposal deltas as a batch operation
```

This eliminates the CPU bottleneck and makes training speed proportional to GPU compute, not CPU iteration.

#### 3b. Hierarchical Diffusion

For generating multi-sentence text, add coarse-to-fine hierarchy:

```
Level 1 (sentences):
  T₁ = 4 steps, corruption at sentence boundaries
  Route graph connects sentence positions
  Output: sentence-level plan

Level 2 (phrases):
  T₂ = 4 steps, corruption at phrase level
  Route graph connects phrase positions, conditioned on Level 1
  Output: phrase-level structure

Level 3 (tokens):
  T₃ = 6 steps, corruption at token level (current system)
  Route graph connects token positions, conditioned on Levels 1-2
  Output: final tokens
```

This is how image diffusion models work (DALL-E 2 generates low-res then super-resolves). Text should work the same way: decide WHAT to say, then HOW to say it, then the exact WORDS.

#### 3c. Scale Parameters

| Parameter | Current | Target | Why |
|-----------|---------|--------|-----|
| hidden_dim d | 256 | 512-1024 | More representational capacity |
| seq_len L | 128 | 512-2048 | Longer coherent output |
| num_blocks | 6 | 12-24 | Deeper model, more abstraction layers |
| Emax | 16 | 32-64 | Richer routing per position |
| vocab_size V | 16,384 | 32,768 | Better tokenization coverage |
| batch_size B | 8 | 32-64 | Smoother gradients |
| Training data | ~1M tokens | Billions | Required for generalization |

Scaling the scorer from 12M to 100-500M parameters requires:
- Moving to multiple GPUs or a single large GPU (A100/H100)
- Gradient checkpointing to fit activations in VRAM
- Mixed precision training (FP16 forward, FP32 accum — Burn supports this)

### Phase 4: Long-Term Architecture Innovations

#### 4a. Multi-Modal Route Graphs

The route graph is modality-agnostic. It maps positions to positions through pheromone-weighted edges. Nothing stops you from having:

```
Positions 0-511:    text tokens
Positions 512-1023: image patch tokens
Positions 1024-1535: audio spectrogram tokens
```

The ants don't know what the tokens represent. They learn which positions help predict other positions. Cross-modal routes (text→image, audio→text) would emerge naturally from pheromone reinforcement.

This could genuinely surpass transformer approaches. Transformers need explicit cross-attention mechanisms between modalities. ERM's ants would discover cross-modal routing automatically.

#### 4b. Semantic Route Graph (Per Gemini Audit)

Add vocabulary-level edges alongside position-level edges:

```
Current: position_5 → position_12  (instance-specific)
Future:  token_"cat" → token_"sat"  (generalizable)
```

This lets the route graph learn dataset-wide patterns like "articles precede nouns" or "adjectives precede the nouns they modify." The per-position graph handles instance-specific routing; the per-token graph handles universal linguistic knowledge.

#### 4c. Distributed Colony

For training on multiple GPUs:

```
GPU 0: scorer forward + backward (data parallel)
GPU 1: scorer forward + backward (data parallel)
CPU: route graph (shared or per-GPU with periodic sync)
```

Pheromone synchronization between graphs: every N steps, average the pheromone values across all GPUs' route graphs. This is analogous to gradient synchronization in distributed SGD but for the discrete routing structure.

---

## Part 5: The Mathematical Formulas, All in One Place

### Corruption Schedule
```
α_t = α_T + (α_1 - α_T) · (T - t) / (T - 1)     mask rate
β_t = β_T + (β_1 - β_T) · (T - t) / (T - 1)     replace rate

Defaults: α_T=0.8, α_1=0.15, β_T=0.1, β_1=0.02, T=6
```

### Diffusion Loss
```
L = (1/T) · Σ_t γ(t) · (1/|C_t|) · Σ_{i ∈ C_t} -log p_θ(x_i | z_t)

where C_t = {i : z_t[i] ≠ x[i]}  (corrupted positions)
      γ(t) = cosine schedule from γ_min=0.5 to γ_max=2.0
```

### Route Aggregation
```
w_raw[b,i,e] = log(φ[b,i,e] + ε) - λ · τ[b,i,e] - μ · age[b,i,e]
w[b,i,:] = softmax(w_raw[b,i,:])    (over Emax; -∞ for empty slots)
r[b,i,:] = Σ_e w[b,i,e] · h[b, src[b,i,e], :]

Defaults: ε=1e-6, λ=1.0, μ=0.01
```

### Pheromone Update Cycle
```
1. Evaporate:     φ_e ← (1 - ρ) · φ_e                              ρ = 0.1
2. Deposit:       φ_e += [η/(1+age)] · tanh(Δ/(σ+ε))               η = 0.5
3. Taint:         τ_e += ζ · max(-Δ, 0)                             ζ = 0.3
4. Taint decay:   τ_e ← (1 - ρ_τ) · τ_e                            ρ_τ = 0.05
5. Age:           age_e += 1
6. Diversity:     if cos_sim(h[src₁], h[src₂]) > 0.9 → φ[weaker] *= 0.8
7. Prune:         remove if φ - λ·τ < -1.0 or age > 1000
```

### Ant Improvement Delta
```
Δ_k = Σ_{i ∈ edits_by_ant_k} [CE(x_i | z_t) - CE(x_i | y_{t-1})]

where CE(x_i | z) = -log softmax(logits(z))[i, x_i]
```

### Fourier Position Embedding
```
For position pos ∈ [0, L) and frequency j ∈ [0, num_freqs):
  v = pos / L
  feature[2j]   = cos(2π · (j+1) · v)
  feature[2j+1] = sin(2π · (j+1) · v)

Then: pos_embed[pos] = Linear(feature)    (learned projection to d dimensions)
```

### Self-Attention (Scorer Internal)
```
Q = h · W_Q,  K = h · W_K,  V = h · W_V     (per head)
A = softmax(Q · Kᵀ / √d_head)
output = A · V
h = h + concat(output_head₁, ..., output_head₄) · W_O    (residual)
```

---

## Part 6: How It All Fits Together — The Full Picture

```
                    ┌─────────────────────────────────────┐
                    │           TRAINING DATA              │
                    │    (tokenized text sequences)        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │      CORRUPTION SCHEDULE             │
                    │   x → z_t (mask + replace)           │
                    │   t: T(heavy) → 1(light)             │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────▼────────────────────────┐
              │              SCORER (GPU)                    │
              │                                              │
              │  Token Embed → Fourier Pos → [FF, Attn] ×3  │
              │  → logits [B,L,V]                            │
              │  → hidden [B,L,d]     → to route graph       │
              │  → uncertainty [B,L]  → to ant policy        │
              │                                              │
              │  Loss: γ(t) · CE(logits, x)[corrupted]       │
              │  Optimizer: Adam (backprop through scorer)    │
              └─────────────┬────────────┬─────────────────┘
                            │            │
                   ┌────────▼───┐  ┌─────▼──────────────────┐
                   │ ROUTE GRAPH │  │    ANT COLONY (CPU)     │
                   │   (CPU)     │  │                         │
                   │ nbr_idx     │  │  256 ants:              │
                   │ phi, taint  │◄─│   26 leaders (explore)  │
                   │ age         │  │  230 followers (exploit) │
                   │             │  │                         │
                   │ Routing:    │  │  Propose edits → y_{t-1}│
                   │ r = Σ w·h   │  │  Compute Δ per ant      │
                   │             │  │  Death if Δ≤0 for K steps│
                   └──────┬──────┘  └──────┬──────────────────┘
                          │                │
                   ┌──────▼────────────────▼──────────────────┐
                   │        PHEROMONE UPDATE (CPU)             │
                   │                                          │
                   │  Evaporate → Deposit → Taint → Diversity │
                   │  → Prune → Age                           │
                   │                                          │
                   │  Signal: Δ (ant improvement)             │
                   │  No gradients. Pure reinforcement.        │
                   └──────────────────────────────────────────┘
```

The two learning systems run in the same loop but are fundamentally different:

| | Scorer (Brain 1) | Route Graph (Brain 2) |
|---|---|---|
| **What it learns** | Token prediction weights | Which positions connect |
| **How it learns** | Gradient descent (Adam) | Pheromone reinforcement |
| **Signal** | ∇L (loss gradient) | Δ (ant improvement) |
| **Space** | Continuous (12M floats) | Discrete (edge topology) |
| **Speed** | Fast convergence | Slow exploration |
| **Strength** | Smooth patterns | Structural patterns |
| **Weakness** | Can't do topology search | Can't do precise weight tuning |

**The hypothesis:** This hybrid produces emergent capabilities that neither system achieves alone. The scorer provides precise predictions within the topology the ants discover. The ants discover topology that enables the scorer to make predictions it couldn't before. They bootstrap each other.

---

## Part 7: Honest Assessment — What's Hard

### 7.1 Training Efficiency

ERM is ~3-4x more expensive per step than an equivalent transformer. The ant evaluation loop is the bottleneck. Until you move ant evaluation to GPU, you'll train slower than a pure gradient descent model of the same size.

**Counterpoint:** ERM may need fewer total parameters to achieve the same quality, because the route graph encodes structural knowledge that would otherwise require billions of weights. This is unproven but is the core bet.

### 7.2 No Convergence Guarantees

Gradient descent has convergence proofs (NTK theory). Ant colony optimization has convergence proofs for combinatorial problems (TSP). Your hybrid has neither. You're in uncharted territory.

**Counterpoint:** GPT-4's training procedure also has no formal convergence guarantees for the quality of generated text. The proof is in the pudding.

### 7.3 The Generalization Gap

Your route graph is per-instance (position-based). It can't generalize across sequences without the semantic routing extension (Phase 4b). Until then, your generalization comes entirely from the scorer's weights and self-attention — exactly like a standard transformer.

**Counterpoint:** The route graph still provides value during training by guiding the scorer's attention patterns. Even if the graph doesn't generalize directly, the scorer weights it helps shape DO generalize.

### 7.4 Scale Is Unknown

Nobody has trained a system like this at scale. The co-evolutionary dynamics between scorer and route graph may exhibit instabilities, oscillations, or plateaus that aren't apparent at 12M parameters.

**Counterpoint:** The only way to find out is to try. The architecture is sound in principle, the code is working, and the near-term improvements are concrete and testable.

---

## Part 8: What To Do Next (Prioritized)

1. **Run the current training with all Phase 1 changes** — measure loss curves with the new Fourier embeddings, self-attention, diversity pressure, etc.

2. **Implement per-position credit assignment** — this is the single highest-impact change for pheromone learning quality.

3. **Batch ant evaluations on GPU** — this is the single highest-impact change for training speed.

4. **Scale to d=512, L=256** — double the model size and sequence length, see if quality scales.

5. **Add spectral corruption schedule** — corrupt rare tokens first at high noise levels.

6. **Train on a large corpus** — the model needs millions of tokens to learn meaningful patterns.

7. **Implement hierarchical diffusion** — required for multi-sentence coherent output.

Each of these is a concrete, implementable step. None requires inventing new math — just applying known techniques to your specific architecture.

---

*This document should be revisited after each training run. The assessment will change as empirical results come in.*
