# ERM: How It Actually Works

A Discord-friendly explainer for the Emergent Route Model.

---

## What is this thing?

ERM is a text model that doesn't use autoregressive generation (no "predict the next token" loop). Instead, it fills in **all positions in parallel** using two cooperating systems:

1. **A GPU transformer scorer** that looks at corrupted text and predicts what the clean text should be
2. **A CPU ant colony** that proposes token edits guided by pheromone trails in a route graph

The scorer learns via normal backprop (AdamW). The route graph learns via ant reward signals (pheromone deposit/evaporation). The two systems are coupled only through logits and improvement deltas. Neither can directly interfere with the other's learning.

This is a **discrete diffusion model** — it denoises text from heavy corruption to clean, step by step — but with ant colony optimization replacing the typical learned reverse process.

---

## The Training Loop (One Iteration)

Each training step runs T=6 noise levels, from heaviest to lightest:

```
for t = 6 down to 1:
    1. Corrupt the clean text at noise level t
       (t=6: 80% masked, t=1: 15% masked)

    2. GPU scorer predicts clean tokens from corrupted input
       → produces logits, uncertainty, hidden states

    3. Compute cross-entropy loss, weighted by γ(t)
       (only on corrupted positions, not the whole sequence)

    4. CPU ant colony reads the logits and proposes edits
       - followers exploit confident, well-routed positions
       - leaders explore uncertain, under-served positions

    5. Merge best proposals → y_new
       (edit budget scales with noise: ~20 edits at t=6, ~3 at t=1)

    6. Second forward pass on y_new (no grad) → compute improvement deltas

    7. Update pheromones based on which ants helped and which didn't

After all 6 levels:
    Backprop the accumulated loss through the scorer (AdamW)
```

The scorer gets better at predicting clean text. The route graph gets better at directing ants toward productive edits. Both improve simultaneously but through completely different mechanisms.

---

## The Scorer Network

A small transformer encoder built in Burn (Rust ML framework). Nothing exotic:

- Token embedding (V+1 entries — the +1 is the MASK sentinel)
- Sinusoidal positional encoding → learned linear projection
- 6 interleaved blocks (alternating feedforward and multi-head attention)
- Two output heads:
  - **Logit head**: linear → [B, L, V] (token predictions)
  - **Uncertainty head**: linear → sigmoid → [B, L, 1] (per-position confidence)

The uncertainty head is what makes the ant colony work. Leaders target high-uncertainty positions. Without it, leaders would just wander randomly.

---

## The Ant Colony

256 ants split into two roles:

### Followers (90% — 230 ants)

Followers exploit. Their position score:
```
score[i] = max_confidence[i] * (total_pheromone[i] + epsilon)
```

High scorer confidence AND strong pheromone routes = high priority. Followers sample positions weighted by this score, then pick tokens from top-k logits with temperature-scaled softmax.

### Leaders (10% — 26 ants)

Leaders explore. Their position score is inverted:
```
score[i] = uncertainty[i] / (total_pheromone[i] + epsilon)
```

High uncertainty AND low pheromone = high priority. Leaders seek out the positions the colony hasn't figured out yet. They also propose **new graph edges** — connecting positions that weren't previously linked.

### Temperature Scheduling (new)

Follower temperature decays over training:
```
T_follower = max(0.3, 1.0 - step/total_steps)
```
Early: T=1.0 (explore broadly). Late: T=0.3 (exploit what works).

Leader temperature scales with uncertainty:
```
T_leader = max(0.5, 2.0 * mean_uncertainty)
```
When the scorer is confused about many positions, leaders get more random. When it's confident, leaders become more targeted.

### Death and Respawn

Each ant tracks consecutive steps where it didn't improve anything. After 5 no-improvement steps, the ant dies and respawns at a random state. This prevents dead-weight ants from accumulating. During early training (warmstart), the death threshold is 4x longer — ants get more patience while the scorer is still learning its first useful representations.

---

## The Route Graph

A sparse directed graph stored as flat arrays: `[B, L, Emax]` where Emax=16.

Each position can have up to 16 incoming edges from other positions. Each edge carries:
- **phi (φ)**: pheromone strength, [0, phi_max]
- **taint (τ)**: penalty for historically harmful routes, [0, tau_max]
- **age**: steps since edge creation

Route weights for attention-style aggregation:
```
w[e] = log(φ + ε) - λ·τ - μ·age
softmax(w) → edge weights
```

The log transform on pheromone means the weight grows sub-linearly. You can't just flood one edge with deposit and dominate — there are diminishing returns. The taint penalty and age decay provide negative signal and mild recency bias.

When a position's 16 slots are full, leader-proposed edges displace the weakest existing edge.

---

## Pheromone Update Pipeline

After each T-step iteration, pheromones go through 7 stages:

**1. Evaporation** — all edges lose 10% of their pheromone. Prevents lock-in.

**2. Deposit (log-bounded)** — this is a recent addition. The deposit function:
```
deposit(Δ) = ln(1 + Δ/σ)
```
where σ is the running standard deviation of positive deltas (tracked by Welford's algorithm).

Why not tanh? Because `tanh(Δ/σ)` saturates to 1.0 for large deltas — a 10x improvement gets the same credit as a 100x improvement. `ln(1 + x)` has better dynamic range: small improvements get proportional credit, large improvements get diminishing but nonzero credit, and it never hits a ceiling.

The deposit is also **per-position**: only edges routing to positions that actually improved receive positive pheromone. An ant that improved 3 out of 8 positions only deposits on edges leading to those 3 positions, not all 8.

**3. Taint deposit** — ants that made things worse accumulate taint on all edges they used. Taint is not age-decayed on deposit — harmful signals stay strong.

**4. Taint decay** — taint fades at 5% per step. Routes that were harmful can recover.

**5-6. Age increment and bounds clamping.**

**7. Diversity pressure** — for each destination, if two incoming edges have cosine similarity > 0.9 (nearly redundant source representations), the weaker one gets penalized by 0.8x. This prevents the graph from collapsing into homogeneous connections. It's the ant-colony analog of Muon's momentum orthogonalization — different routes should carry different information.

---

## Pheromone Rescaling (MuonClip Analog)

New addition. After each iteration, if `max(φ) > 0.8 * phi_max`:
```
γ = sqrt(threshold / max_phi)
φ[e] *= γ   for all edges
```

This is the direct analog of QK-Clip from the Muon optimizer paper. In Muon, attention logits that grow too large collapse the softmax into a near-hard argmax, killing gradient flow. Same problem here: if one edge's pheromone dominates, the route softmax collapses and the colony can't explore alternatives. The sqrt rescaling damps the maximum without zeroing out smaller values.

---

## Spectral Corruption Schedule (new)

Standard corruption treats all positions equally — each has the same probability of being masked. The spectral variant biases corruption based on per-position surprisal (cross-entropy from a previous forward pass):

```
blend = (t-1) / (T-1)     // 0 at t=1 (lightest noise), 1 at t=T (heaviest)

priority[i] = blend * surprisal[i]
            + (1-blend) * (max_surprisal - surprisal[i])
```

Normalize so mean priority = 1.0, then multiply each position's mask/replace rates by its priority.

The effect:
- **At t=T (heavy noise)**: rare, surprising positions get corrupted more. The model trains on hard cases under heavy noise.
- **At t=1 (light noise)**: common, expected positions get corrupted more. The model fine-tunes on easy cases under light noise.

This is a frequency-aware curriculum — analogous to how Fourier features let MLPs learn high-frequency functions first at coarse resolution, then refine low frequencies. The corruption schedule biases *which information* gets corrupted at each noise level, not just *how much*.

---

## What's NOT Here

Honest accounting of what ERM doesn't have (yet):

- **No gradient signal to the colony.** Pheromone credit is a reward signal, not a gradient. The graph can't do anything as precise as backprop. This limits how fast the routing can improve.
- **No hierarchical diffusion.** The T=6 levels are flat — each level runs the same-sized colony and scorer. A hierarchical approach (coarse scorer at t=6, fine scorer at t=1) could be more efficient.
- **No batch ant evaluation on GPU.** Colony proposals are evaluated with a second forward pass, but the proposal generation itself is CPU-bound. Moving position scoring to GPU would help at scale.
- **Vocab size vs. speed tradeoff.** With the full 50K BPE vocab, wgpu JIT compilation takes minutes on first step. The scorer's embedding table and logit head are proportionally larger.

---

## How Does Inference Work?

Start with a fully masked sequence (optionally with a fixed prompt prefix). Then iterate K steps:

```
for step = 0 to K-1:
    1. Forward pass → logits for all masked positions
    2. Compute confidence = max(softmax(logits)) at each masked position
    3. Fill the top-N most confident positions with their argmax token
       (N scales down: many fills early, few fills late)
    4. Remaining positions stay masked for next iteration
```

No autoregressive left-to-right constraint. The model fills whatever it's most confident about first, regardless of position. A sentence might get its verbs filled first, then nouns, then articles — whatever the scorer is surest about at each step.

---

## The Two-Loop Architecture (Why This Matters)

Most neural networks have one optimization loop: gradient descent on parameters. ERM has two:

| | Scorer (gradient) | Route graph (pheromone) |
|---|---|---|
| **What learns** | Weight matrices | Edge strengths |
| **Signal** | Cross-entropy gradient | Ant improvement deltas |
| **Update rule** | AdamW | Evaporate → deposit → taint |
| **Hardware** | GPU | CPU |
| **Precision** | Exact (chain rule) | Approximate (reward) |
| **What it controls** | Token predictions | Which positions get edited |

The scorer answers "what should each token be?" The colony answers "which tokens should we try to change, and how?" These are different questions, and they're answered by different optimization processes running on different hardware at different speeds.

The pheromone system is the colony's long-term memory. Without it, each step's ants would start from scratch. With it, successful routing patterns accumulate across steps, and the colony develops persistent structure — routes that reliably guide edits toward productive positions.

This is emergent routing. Nobody designed the routes. They grow from the colony's cumulative experience, shaped by the same forces that shape real ant trails: positive feedback (deposit), negative feedback (taint), decay (evaporation), exploration (leaders), and exploitation (followers).

---

## Current Training Output

Here's what a training step looks like:

```
[diffusion step     25] loss=6.4559 lr=0.000500 f_temp=1.00 l_temp=1.76
                        edits=327 mean_phi=0.0071 deaths=238 pruned=134 inserted=96
```

- `loss`: diffusion cross-entropy, averaged over T levels
- `lr`: scorer learning rate (AdamW)
- `f_temp`: follower temperature (decays 1.0 → 0.3 over training)
- `l_temp`: leader temperature (scales with mean uncertainty)
- `edits`: total colony edits across all T levels
- `mean_phi`: average pheromone across all edges (starts near phi_init=0.05)
- `deaths`: ants that hit their no-improvement streak limit
- `pruned`: edges removed (too weak or too old)
- `inserted`: new edges proposed by leaders
