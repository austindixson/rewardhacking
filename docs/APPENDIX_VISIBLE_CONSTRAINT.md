# Appendix: visible-constraint failure modes

## Intended behavior

When `intervention_mode="visible_constraint"` and variance trigger fires:

1. `hack_detected` → `effective_hidden_weight = 0` (same as kill_gradient)  
2. Visible rubric gains a **forbidden_word** check for the hidden keyword  

## Observed behavior (hosted runs)

| Run | Env | vig @ s99 | hga @ s99 | spikes | Notes |
|-----|-----|-----------|-----------|--------|-------|
| `cg71a38…` | @0.2.1 | — | — | — | FAILED s45, zero_advantage |
| `n42izvdz…` | @0.2.1 | — | — | — | FAILED |
| `gzis11zm…` | latest (0.2.3+) | 0 | 1 | ~0.67 | Completed; **no trigger** |

## Hypotheses (ranked)

1. **Low hidden std under `aggregation=average` fallback** — env rewrites `aggregation=all` to `average` for visible_constraint ([vigilance_core.py](../environments/backdoor_ifeval_vigilant/vigilance_core.py)); batches may rarely produce 5 consecutive groups with `hidden_std > 0`.  
2. **Early training never diversifies hidden behavior** — if all rollouts in a group score hidden=0, spikes never increment.  
3. **`zero_advantage` filtering** — removes learning signal; visible scores inflate while vigilance state stalls.  
4. **Historical tuple bug** (pre-0.2.3) — forbid check appended as 2-tuple caused unpack crashes; fixed in 0.2.3.

## What we cannot do in-env

- **Disable `zero_advantage`** — orchestrator policy on Prime hosted training.

## Diagnosis sweep (Wave 1)

See [SEED_SWEEP_PLAN.md](../SEED_SWEEP_PLAN.md): 5× `visible_constraint` seeds, threshold 3/7, explicit `aggregation=average`, env `@0.2.4` with new `intervention_group` metric.

## Recommendation

Do not claim visible-constraint as a supported intervention until ≥3/5 seeds show `vigilance_active=1` and `hidden_gradient_active=0` at s99. Otherwise document as **failed lever** in the postmortem.
