# Phase 2A — Binary regret summary

**Trigger step:** first `vigilance_active ≥ 0.5`, or partial gradient off with vigilance on (see `scripts/metrics_timeline.py:first_regret_trigger_step`).

**Regret at trigger:** `control_visible[t] − method_visible[t]` at that step.

| Config | Trigger | Vis@trigger | Ctrl@trigger | Regret@trigger | Mean vis post | Mean regret post | Cum. deficit | s99 vis | Δs99 vs ctrl |
|--------|---------|-------------|--------------|----------------|---------------|------------------|--------------|---------|--------------|
| control | — | — | — | — | — | — | — | 0.662 | +0.000 |
| vigilant | 22 | 0.146 | 0.219 | 0.073 | 0.330 | 0.119 | 9.973 | 0.669 | +0.006 |
| p3-vigilant | 14 | 0.175 | 0.090 | -0.085 | 0.328 | 0.092 | 8.341 | 0.591 | -0.072 |
| random | 1 | 0.033 | 0.013 | -0.021 | 0.413 | -0.041 | 2.363 | 0.673 | +0.010 |
| oracle | 1 | 0.075 | 0.013 | -0.062 | 0.346 | 0.026 | 5.552 | 0.842 | +0.179 |
| behavior-penalty | 18 | 0.138 | 0.133 | -0.005 | 0.583 | -0.149 | 0.310 | 0.887 | +0.225 |

**Read:** Oracle/random kill early (step ~1) with small instantaneous regret but very different post-kill learning; variance vigilant triggers late (~step 23) with similar small regret at trigger yet s99 ≈ control. Behavior-penalty wins on s99 without relying on gradient kill timing.
