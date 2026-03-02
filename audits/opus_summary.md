# Opus Audit — ERM Route Graph & Colony Training (Summary)

Date: 2026-03-02
Agent: Opus (claude-opus-4-6)

## Executive summary
- Critical bug: the route aggregation receives a zero hidden state (`hidden = vec![0.0...]`) — the graph has no semantic signal. This disconnect explains most poor behaviors (seeded edges pruned, random proposals dominate).
- Pheromone deposit uses `η·tanh(relu(Δ))`, which saturates quickly; large Δ values collapse to nearly the same deposit and lose ranking information.
- Death/respawn is too aggressive early in training — very high ant death rates cause edge sparsity and unstable learning.
- Batch-aggregation bug: ant deltas were computed only from the first batch item (batch_idx=0), discarding the rest of the batch's signal.
- Implementation issues: phi initialized on empty slots, redundant tensor cloning (GPU↔CPU transfers), a dead/ineffective loop in pheromone update.

## Top recommendations (priority order)
1. Fix route aggregate: pass real encoder hidden states into `route_aggregate` immediately. (P0)
2. Aggregate ant deltas across the full batch before pheromone updates. (P1)
3. Ensure empty edge slots have phi=0; reserve phi_init for real seeded/inserted edges. (P1)
4. Reduce death aggressiveness for warm-start (increase `death_streak`) and consider respawn-from-pool policy. (P2)
5. Replace tanh(relu(Δ)) with a normalized mapping (e.g., `tanh(Δ / σ)` or softplus) to preserve dynamic range. (P2)
6. Reduce unnecessary tensor cloning; streamline the GPU↔CPU bridge to lower memory pressure. (P3)

## Evidence & artifacts
- Snapshot used: `data/colony-checkpoints/snapshots/step_00100.json` (shows seeded edges collapsed to ~114, heavy pruning)
- Audit JSON: `audits/opus_audit.json`
- Gemini-style audit: `audits/gemini_audit.json`
- Preliminary consensus: `audits/consensus_prelim.json`

## Next suggested steps
- Implement (1) and (2) as code fixes, run a short 200-step CPU smoke test, then resume GPU training.
- While fixing, run the validation experiments from the audits (route_aggregate assertion, tanh scaling plot, death ablation).

---

Files written:
- JSON audit: `./audits/opus_audit.json`
- Summary markdown: `./audits/opus_summary.md`
- Gemini-style JSON: `./audits/gemini_audit.json`
- Consensus: `./audits/consensus_prelim.json`

