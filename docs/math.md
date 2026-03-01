# ERM Math Reference

Quick reference for objectives, update rules, and schedules.

## Noise Schedule

Linear schedule over T refinement steps:

```
α_t = α_T + (α_1 - α_T) * (T - t) / (T - 1)    # mask rate
β_t = β_T + (β_1 - β_T) * (T - t) / (T - 1)    # replace rate
```

Default values:
- `α_T = 0.8`, `α_1 = 0.15`
- `β_T = 0.1`, `β_1 = 0.02`
- `T = 6`

## Corruption Operator

For each token position `i`:
1. With prob `α_t`: set `y_t[i] = MASK`
2. Else with prob `β_t`: set `y_t[i] = confuser` (sampled from unigram)
3. Else: keep `y_t[i] = x[i]`

## Denoising Loss

```
L_denoise = (1 / |corrupted|) * Σ_{i: y_t[i] != x[i]} CE(softmax(logits[i]), x[i])
```

## Route Aggregation

For each destination position `i`:
```
w_raw[b,i,e] = log(φ[b,i,e] + ε) - λ * τ[b,i,e] - μ * age[b,i,e]
w[b,i,:]     = softmax(w_raw[b,i,:])    # over Emax dimension; -inf for empty slots
r[b,i,:]     = Σ_e w[b,i,e] * h[b, nbr[b,i,e], :]
```
Hyperparams: `ε=1e-6`, `λ=1.0`, `μ=0.01`

## Improvement Metric

```
Δ_i = CE_before(i) - CE_after(i)    for edited position i
Δ_k = Σ_{i ∈ edits_by_ant_k} Δ_i   per-ant improvement
```

## Pheromone Updates

Applied in order after each step:

**Evaporation:**
```
φ_e ← (1 - ρ) * φ_e
```

**Deposit** (for each ant k that used edge e):
```
φ_e ← φ_e + η * relu(Δ_k)
```

**Taint update:**
```
τ_e ← clamp(τ_e + ζ * relu(-Δ_k), 0, τ_max)
τ_e ← (1 - ρ_τ) * τ_e
```

Default hyperparams: `ρ=0.05`, `η=0.01`, `ζ=0.05`, `τ_max=5.0`, `ρ_τ=0.01`

## Leader Utility (EMA)

```
U(e) ← (1 - γ) * U(e) + γ * relu(Δ_when_used)
```
`γ = 0.9` (EMA decay)

## Ant Position Scores

**Followers** — exploit:
```
score_i = editable[i] * conf_i * route_strength_i
conf_i  = max(softmax(logits[i]))
route_strength_i = Σ_e φ[i,e]
```

**Leaders** — explore:
```
score_i = editable[i] * u_i * (1 / (route_strength_i + ε))
u_i     = entropy(softmax(logits[i]))   # or learned uncertainty head
```
