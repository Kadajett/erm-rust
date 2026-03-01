# ERM vs PCN: Comparison Log

Running comparison between ERM (Emergent Route Model) and PCN (Predictive Coding Network).  
Both trained on the same dataset with the same tokenizer.

## Methodology

- **Dataset:** TBD — agree with rust-pcn before Phase 1 begins
- **Tokenizer:** TBD — target same 16k BPE vocab
- **Train/val/test splits:** TBD — document exact splits and seed
- **Eval protocol:** masked-token accuracy on same held-out set; generation perplexity under same reference LM
- **Compute budget:** same number of gradient steps (TBD) or same wall-clock time

## Results Table (fill in as milestones complete)

### Milestone 1 (Baseline Denoiser)

| Metric | ERM | PCN | Notes |
|---|---|---|---|
| Masked-token accuracy (val) | — | — | |
| Training loss @ 10k steps | — | — | |
| VRAM peak (B=8, L=128) | — | — | |
| Parameters | — | — | |

### Milestone 3 (Colony Refinement / PCN Equivalent)

| Metric | ERM | PCN | Notes |
|---|---|---|---|
| 1-step refinement CE improvement | — | — | |
| Training throughput (steps/sec) | — | — | |

### Milestone 6 (Full Generation)

| Metric | ERM | PCN | Notes |
|---|---|---|---|
| Generation perplexity (from mask) | — | — | |
| Convergence step (val loss plateau) | — | — | |
| Sample quality (subjective, 10 samples) | — | — | |

## Hypotheses

| Hypothesis | Status | Evidence |
|---|---|---|
| H1: ERM matches PCN accuracy with fewer params (graph compensates) | untested | — |
| H2: ERM slower early convergence, better late convergence | untested | — |
| H3: ERM shows more local coherence in output text | untested | — |
| H4: PCN converges faster per step; ERM more VRAM-efficient at equal quality | untested | — |

## Qualitative Observations

*(Fill in as training progresses.)*

---

*Updated: 2026-03-01*
