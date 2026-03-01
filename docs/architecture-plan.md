# Untitled

Prompt for agent: ERM (Emergent Route Model) architecture, math, training, and build plan

You are designing ERM: Emergent Route Model, a non-autoregressive, non-diffusion text generator that learns via stigmergic exploration and exploitation. The model maintains a persistent route graph (pheromone memory) that evolves during training and inference. A small neural scorer proposes edits to a partially corrupted sequence. Two ant populations operate per refinement step:

Leaders (10%) explore new routes and propose risky edits and new edges.

Followers (90%) exploit existing routes and propose conservative edits along strong edges.

Any ant that fails to improve the objective for K actions “dies”, taints its traversed routes, and is replaced.

The model generates text by parallel refinement over T steps: initialize a masked or noisy sequence, then iteratively fill and correct tokens until convergence.

Your deliverables:

1. A precise architecture spec: modules, tensor shapes, forward pass, route graph representation, ant policies, edit selection, and how the route graph grows and is pruned.
2. The math: objective, route reinforcement update rules, leader exploration credit, tainting, and the discrete noise schedule.
3. A complete training recipe: data pipeline, batching, schedule, hyperparameters, stability tricks, and evaluation.
4. An inference recipe: how to generate from scratch, how to condition on a prompt, and how to stop.
5. A project plan in milestones with acceptance criteria and test strategy.
6. Estimated data shapes and memory layout for Rust + burn with CUDA, including sparse graph storage and batched ant proposals.

Constraints

Single consumer GPU: RTX 3050 (limited VRAM).

Implementable in burn (Rust) with CUDA backend.

Must be parallelizable and not require huge Transformers.

Must include output tensor shapes, and how length is handled.

Must explain “network shape”: what neural network exists, and what part is dynamic memory.

---

ERM: Architecture specification

A. Data representation

Tokenization

Vocabulary size V (BPE). Typical: 16k to 50k.

Sequence length L: start with 128 or 256 for feasibility. Later grow to 512.

State at refinement step t

Ground truth tokens x ∈ [0..V-1]^L

Corrupted / current tokens y_t ∈ [0..V]^L where V denotes MASK sentinel (or reserve an id)

Optional: store corruption mask m_t ∈ {0,1}^L indicating which positions are “unknown/noisy”.

Conditioning

For prompted generation:

Split into prefix p and generated region g.

Prefix positions are clamped (not editable), generated positions are editable.

Use an edit mask editable[i].

---

B. Core components

ERM has two “brains”:

1. Neural Scorer Network Sθ: a lightweight sequence model that outputs per-position candidate tokens and confidence, plus routing features.
2. Route Graph Memory G: a dynamic sparse directed graph storing pheromone trails, taint, and usage stats.

B1. Neural Scorer Sθ

Purpose: given y_t and a route-conditioned message, propose distributions over tokens for each position, plus an uncertainty signal.

A feasible design on 3050 is a small Transformer-like encoder (or Mamba-ish), but you must keep it small and offload “long-range structure” into the route graph.

Recommended “network shape”:

Token embedding: E ∈ R^{V × d}

Position embedding: P ∈ R^{L × d} (or rotary)

B blocks of lightweight self-attention OR gated conv/Mamba

Route-conditioned cross message: a per-position vector r_i ∈ R^d computed from G and current tokens

Output heads:

Token logits: logits ∈ R^{Bsz × L × V}

Uncertainty: u ∈ R^{Bsz × L} (for leader targeting)

Optional length head: len_logits ∈ R^{Bsz × Lmax}

Concrete hyperparameters to start:

d = 256

B = 6 blocks

heads h = 4 if attention is used

MLP expansion 4d This is intentionally “small LM sized”.

Route-conditioned input

Instead of full attention everywhere, you compute a route message:

r = RouteAggregate(G, y_t) with shape Bsz × L × d Then scorer consumes emb(y_t) + pos + r.

This makes the dynamic memory actually matter.

B2. Route Graph Memory G

ERM uses a typed sparse graph that connects token positions to token positions, plus optional latent “concept nodes”.

Minimum viable node set:

Position nodes: i = 0..L-1 (per sample) Optionally:

Concept nodes: ids for token clusters or n-grams

Global nodes: “sentence intent” or “topic” for longer contexts

Edge structure:

Directed edge (j → i) means “position j provides useful evidence for predicting i”

Each edge stores:

pheromone φ ∈ R+

taint τ ∈ R+

cost c (optional, for pruning)

age a and usage count n

optional type id (local, long-range, concept)

Because this must fit on GPU, you must keep it sparse:

Max edges per node: Emax (like 8, 16, or 32)

Graph per sample, per batch, or a hybrid (see below)

Storage layout suggestion for burn:

CSR-like:

row_ptr: [L+1] i32

col_idx: [E] i32

phi: [E] f16 or f32

taint: [E] f16 or f32

age: [E] i16

etype: [E] u8

Compute route message:

For each destination i, aggregate sources j:

r_i = Σ_{(j→i) in G} w_{j→i} * h_j Where:

h_j is the current hidden embedding at j (from token embedding or last block)

w = softmax( log(φ + eps) - λτ - μage ) over incoming edges to i

This is like sparse attention whose connectivity is learned via pheromones.

---

C. Ant populations and actions

At each refinement step, you simulate A ants per sequence.

Leaders count A_L = ceil(0.1A)

Followers count A_F = A - A_L

Each ant k produces an edit proposal set:

Choose a subset of positions P_k (size p, like 1..8)

For each i in P_k, propose token t_i and confidence q_i

Also produce an edge usage trace: which edges in G were used to justify edits

How ants choose positions

Followers:

Sample positions with high pheromone support and high model confidence.

Score per position:

score_i = (1 - editable[i]?) * 0 + editable[i] * (conf_i) * (route_strength_i) Leaders:

Sample high uncertainty positions or low pheromone areas:

score_i = editable[i] * (u_i) * (1 / (route_strength_i + eps))

Where:

conf_i = max_softmax(logits[i])

u_i could be entropy of logits or a learned head

route_strength_i could be sum of incoming φ

How ants propose tokens

From scorer logits:

Take top-k candidate tokens per position.

Followers sample from top-k with temperature < 1.

Leaders sample with temperature > 1, or include a novelty bonus.

---

D. Edit application: conflict-free merge

Many ants propose edits. You need a deterministic merge:

1. Collect all candidate edits (i, token, predicted_gain, ant_type).
2. For each position i, keep the best candidate by predicted_gain.
3. Apply a constraint:

Max edits per step M (like 0.15L).

Do not edit clamped prefix tokens.

1. Produce y_{t-1}.

This makes each refinement step stable and GPU-friendly.

---

ERM: Math and training objective

1. Discrete corruption process (noise schedule)

Define a schedule q_t(y_t | x) for t=1..T.

Example:

At t=T, corrupt heavily:

mask rate α_T = 0.8

replace rate β_T = 0.1

At t=1, corrupt lightly:

α_1 = 0.15

β_1 = 0.02 Optionally a linear schedule from T to 1.

Corruption operator:

With prob α_t: set token to MASK.

Else with prob β_t: replace token with a confuser token sampled from:

unigram distribution, or

top frequent tokens, or

same token class buckets

Else keep original.

Training sample includes multiple t values (random t per batch or multi-t).

1. Denoising loss (supervised backbone)

Given (x, y_t), scorer predicts logits for each position:

pθ(i) = softmax(logits[i])

Compute cross-entropy only on corrupted positions:

L_denoise = Σ_{i: y_t[i] != x[i]} CE(pθ(i), x[i])

This anchors learning in standard likelihood terms, even though inference is not autoregressive.

1. Colony improvement signal (credit assignment)

When applying edits and producing y_{t-1}, define an improvement metric:

Fast local metric:

Δ = Σ_{i in edited} [ log pθ(x_i | y_{t-1}, G) - log pθ(x_i | y_t, G) ]

Cheaper alternative:

Δ ≈ Σ_{i in edited} [ CE_before(i) - CE_after(i) ]

This is computable in one additional forward pass, or approximated via logits deltas.

1. Pheromone update (stigmergic learning)

Each ant produces a trace of edges E_k it relied on. Update per edge e:

Evaporation:

φ_e ← (1 - ρ) * φ_e

Deposit:

φ_e ← φ_e + η * relu(Δ_k)

Taint:

τ_e ← clamp( τ_e + ζ * relu(-Δ_k), 0, τ_max )

Optionally decay taint too:

τ_e ← (1 - ρ_τ) * τ_e

So “bad routes” get temporarily suppressed.

1. Leader exploration reward (delayed utility)

When a leader proposes a new edge e_new, assign it a “provisional pheromone” and track utility:

Maintain U(e) as EMA of positive contributions over next N steps:

U(e) ← (1-γ)U(e) + γ * relu(Δ_when_used) Leaders receive reward proportional to U of edges they introduced:

R_leader = Σ_{e introduced by leader} U(e)

If you want a policy gradient style:

L_policy = - E[ R_leader * log πθ(action) ] But you can start without explicit RL by making leader behavior heuristic and only letting the graph update be the learning signal.

1. Death and replacement

Each ant has a “streak” counter s:

If Δ_k <= ε then s += 1 else s = 0

If s >= K then ant dies:

apply taint to its recently used edges

respawn ant with new seed and type distribution fixed (10/90)

This enforces continual exploration and reduces drift into bad local minima.

---

Output shapes and key tensors

Let:

B batch size

L sequence length

V vocab size

d hidden dim

A total ants per sequence

k top-k candidates per position

Minimum tensors each step:

Tokens

y_t: [B, L] int32

editable: [B, L] bool (optional)

Scorer forward

emb: [B, L, d] f16

route_msg r: [B, L, d] f16

hidden: [B, L, d] f16

logits: [B, L, V] f16 or f32 (often f16 storage, f32 softmax)

uncertainty u: [B, L] f16

Top-k

topk_ids: [B, L, k] int32

topk_scores: [B, L, k] f16

Ant proposals (batched)

Represent as fixed maximum proposals per ant pmax:

ant_pos: [B, A, pmax] int32 (positions)

ant_tok: [B, A, pmax] int32 (token ids)

ant_gain: [B, A, pmax] f16 (predicted gain)

You then reduce to per-position best candidate:

best_tok: [B, L] int32 (or sentinel for none)

best_gain: [B, L] f16

Apply up to M edits:

y_{t-1}: [B, L] int32

Route graph (CSR)

For each sample, store incoming edges (destination-centric is convenient):

row_ptr: [B, L+1] int32

col_idx: [B, E] int32

phi: [B, E] f16

taint: [B, E] f16

age: [B, E] int16

etype: [B, E] u8

With E = L * Emax approximately.

---

Growth potential and pruning

ERM “grows the network” primarily via graph structure, not by increasing NN parameters.

Growth axis:

edges per node (Emax)

concept nodes (optional)

per-edge features

You must bound growth:

hard cap Emax per destination node

pruning policy every N steps:

drop lowest φ - λτ edges

drop oldest unused edges

optional “route consolidation”:

if many edges originate from tokens with same embedding cluster, replace with a concept node edge

This yields scaling without parameter bloat, while keeping GPU predictable.

---

Training recipe

Data pipeline

Use plain text dataset, tokenize into sequences of length L with stride.

For each batch:

sample a random t in 1..T

corrupt x to y_t using α_t and β_t

set editable mask (all editable for denoising; for prompt-conditioning, prefix fixed)

Inner loop per batch

For each refinement step (could be 1 step during training, or multi-step unrolling later):

1. Compute r = RouteAggregate(G, y_t)
2. Forward scorer: logits, uncertainty, and optionally route features
3. Compute supervised L_denoise
4. Build top-k candidates
5. Sample ants:

followers pick positions from high route_strength and high confidence

leaders pick high uncertainty and low route_strength

1. Ant proposals -> merge -> y_{t-1}
2. Compute improvement Δ (local CE delta)
3. Update pheromones and taints based on edge usage traces
4. Optionally update leader utility tracking

Stability tricks

Start with no leaders for the first phase, or leaders that only propose edits but not edges.

Warm-start the scorer as a plain denoiser for a while.

Introduce pheromone updates gradually (small η).

Keep route aggregation differentiable but updates non-differentiable (that’s fine, it’s a memory system).

Evaluation

Denoising accuracy on masked tokens

Generation quality:

run T refinement steps from all-mask and compute perplexity of final output under a reference LM, or do human eval

Route metrics:

average φ distribution entropy

prune rate

fraction of tainted edges

---

Inference recipe

Unconditional generation

1. Pick L (or sample length from a small length model).
2. Initialize y_T = [MASK]*L.
3. For t = T..1:

scorer -> ants -> apply edits

update graph optionally (online inference learning can be disabled)

1. Stop when:

no edits applied, or

mean confidence above threshold, or

max steps reached

Prompted generation

1. Set prefix tokens fixed.
2. Initialize suffix as MASK.
3. Same refinement steps but editable mask blocks prefix edits.

---

Implementation plan (milestones, acceptance criteria, tests)

Milestone 1: Baseline discrete denoiser in burn

Deliver:

Tokenizer integration and dataset loader

Corruption schedule q_t

Small scorer network producing logits [B, L, V] Acceptance:

Training decreases masked-token CE reliably

Unit tests for corruption and masking Tests:

Golden corruption tests

Shape assertions on forward pass

Milestone 2: Route graph storage + RouteAggregate kernel

Deliver:

CSR storage per sample

RouteAggregate computes r [B, L, d] from embeddings h [B, L, d] Acceptance:

RouteAggregate matches CPU reference within tolerance

Performance: stable and does not blow VRAM Tests:

Random CSR property tests

Deterministic small-case comparisons

Milestone 3: Followers only colony refinement (no graph updates yet)

Deliver:

top-k extraction

follower ant sampler

merge and apply edits Acceptance:

1-step refinement improves CE vs no refinement on average

Deterministic merge behavior Tests:

Conflict resolution tests

“No-edit” stability tests

Milestone 4: Pheromone update loop (stigmergy)

Deliver:

edge usage tracing from RouteAggregate

pheromone evaporation, deposit, and taint updates

pruning to enforce Emax Acceptance:

Edges concentrate on useful routes (measurable via higher Δ over time)

No unbounded graph growth Tests:

Update invariants: φ non-negative, τ in bounds

Pruning preserves CSR integrity

Milestone 5: Leaders + exploration mechanics + death/respawn

Deliver:

leader position sampler uses uncertainty and low route strength

leader edge proposal mechanism

death/taint streak logic Acceptance:

Leaders discover new edges that become used by followers later (utility > baseline)

Training remains stable Tests:

Reproducibility under fixed seed

Edge proposal respects caps and does not corrupt CSR

Milestone 6: Multi-step unrolled refinement training

Deliver:

unroll 2..T steps during training (truncated)

optional stop criteria Acceptance:

Generation from all-mask produces coherent text better than baseline denoiser Tests:

Regression tests on fixed prompts

Memory leak and VRAM stability tests

Milestone 7: End-to-end CLI and benchmarks

Deliver:

CLI: train, eval, generate

benchmarks: throughput (tokens per sec), VRAM, refinement steps Acceptance:

Produces outputs consistently, bounded latency Tests:

Snapshot tests of generated samples (with seed)

---

Rust + burn CUDA: estimated shapes and memory notes

Recommended starting config

L = 128

B = 8 (tune with VRAM)

V = 32k (or 16k to start)

d = 256

B_blocks = 6

T = 6 refinement steps

Emax = 16

A = 256 ants per sequence

k = 8 top-k

Memory roughness (ballpark, not guaranteed)

Biggest tensor is logits: [B, L, V]

8 * 128 * 32000 = 32,768,000 elements

fp16 is ~64 MB just for logits So you may need:

smaller V (16k)

smaller B

compute logits in chunks (blockwise vocab projection)

or use tied embeddings with sampled softmax

Pragmatic approach on 3050:

Start with V=16k or use adaptive/sampled softmax.

Or project to a smaller candidate set per step using top-k from a smaller head.

burn implementation hints

Implement RouteAggregate as:

gather source embeddings by col_idx

compute weights from phi, taint, age

segment-softmax per row (destination)

weighted sum If burn lacks segmented ops you may need a custom CUDA kernel or approximate:

fixed Emax per node, store as dense [B, L, Emax] neighbors Then RouteAggregate is just dense ops:

nbr_idx: [B, L, Emax]

phi: [B, L, Emax]

taint: [B, L, Emax] This is often easier on GPU than CSR and still bounded.

Dense-neighbor format recommended for v1:

nbr_idx: i32 [B, L, Emax]

phi: f16 [B, L, Emax]

taint: f16 [B, L, Emax] RouteAggregate:

h_nbr: [B, L, Emax, d] gathered

w: [B, L, Emax] softmax over Emax

r: [B, L, d] = Σ w * h_nbr

This avoids segmented softmax complexity.

---

What is the “shape of the NN” exactly?

ERM’s learned parameters are the scorer Sθ:

Embedding table [V, d]

B blocks of either:

attention weights (QKV projections) and MLPs, or

gated conv / SSM style blocks

Output projection [d, V] (often tied to embeddings)

So it looks like a small encoder-only network that predicts masked tokens.

What makes ERM different is that the effective computation graph is shaped by the evolving route memory G:

G controls sparse long-range information flow (like adaptive attention topology).

Leaders alter G by proposing new edges.

Followers exploit G by repeatedly using strong routes.

So ERM scales by growing and refining G, not by scaling θ.

For CPU optimization on communicating with the GPU.

For CPU to GPU efficiency, treat ERM like a streaming, structure of arrays pipeline where the CPU only ever emits tiny, contiguous, fixed-shape control buffers and the GPU owns all heavy state. Concretely: keep the entire y_t [B,L], route neighbor lists nbr_idx [B,L,Emax], pheromone fields phi/taint [B,L,Emax] (FP16), and scorer weights resident on GPU for the full run; on the CPU side, build only (a) the next minibatch of token ids and corruption masks and (b) a compact “command buffer” that encodes ant seeds and any rare structural mutations (edge insertions, prunes) as fixed-width records. Use pinned host memory plus double or triple buffering so CPU can fill buffer N+1 while GPU consumes buffer N, and launch a single fused kernel per refinement step that does: route aggregation (gather plus softmax over Emax), scorer forward (FP16 with FP32 accum), top-k extraction, ant sampling, conflict-free merge, and pheromone updates, writing back y_{t-1} in place so no round trips occur. For L1 friendliness on the i7, use SoA contiguous arrays aligned to cache lines: u16 tokens[B*L], u16 editable[B*L], i32 nbr_idx[B*L*Emax], f16 phi[B*L*Emax], f16 taint[B*L*Emax], and keep any CPU-side preprocessing strictly linear scans (no pointer chasing, no hash maps). If you must do pruning or leader edge proposals on CPU, do it as a batched “compaction pass” over fixed-size neighbor slots (swap-remove inside [Emax] per destination) so memory access stays sequential, then upload only the modified slices (use a dirty-bit range list per batch item to avoid copying the whole graph). The meta-goal is: CPU touches only compact contiguous buffers sized O(B·L) per step, while GPU touches O(B·L·Emax·d) and everything stays FP16 on device with minimal synchronization points.