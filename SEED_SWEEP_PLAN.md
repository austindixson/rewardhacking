# Seed sweep plan (postmortem diagnosis)

**Goal:** Quantify instability and diagnose `visible-constraint` before writing the negative-result post.

## Constraints (read first)

| Constraint | Implication |
|------------|-------------|
| Prime free tier **2 concurrent** runs | Queue sweeps in waves; ~1 h/run |
| **Max 10 queued** runs | `vig-p3-seed-01`…`05` **queued** 2026-05-24 (~11 ahead) |
| **`zero_advantage` filter** | Orchestrator-level — **cannot disable** from env. Mitigate with `aggregation=average` or accept high filter rates. |
| No `seed` in TOML | Each hosted run = independent implicit seed (valid for spread) |
| Env **v0.2.4+** | Exports `intervention_group` for trigger-step figures |

## Wave 1 (launch first) — 13 runs

### A. visible-constraint (5 seeds)

| Config | Purpose |
|--------|---------|
| `configs/sweeps/vc-seed-01.toml` … `05` | Default: `visible_constraint`, hub `@0.2.4`, agg=all→average fallback |

### B. visible-constraint variants (3)

| Config | Purpose |
|--------|---------|
| `vc-threshold-3.toml` | `spike_threshold=3` |
| `vc-threshold-7.toml` | `spike_threshold=7` |
| `vc-agg-average.toml` | Explicit `aggregation=average` |

### C. Vigilant replicability (5 + 5)

| Config | Purpose |
|--------|---------|
| `vig-p0-seed-01.toml` … `05` | `vigilant-early-warning.toml` @0.2.0 |
| `vig-p3-seed-01.toml` … `05` | `p3-vigilant-replicate.toml` @0.2.0 |

## Wave 2 (optional, if Wave 1 inconclusive)

- Expand to 8–10 seeds per arm (add `vc-seed-06` … `10`, etc.)
- `vc-kill-gradient.toml` — same vigilance path but `intervention_mode=kill_gradient` (isolates constraint effect)
- Pin `@0.2.0` + `visible_constraint` only for historical reproduction (tuple bug fixed in 0.2.3+)

## Per-run logging

After each run:

```bash
prime train metrics <RUN_ID> --plain --min-step 0 --max-step 99
python scripts/fetch_metrics.py --run-id <RUN_ID>
```

Record: s99 visible/hidden/hga/vig_active, `zero_advantage_frac`, max `intervention_group` on timeline.

## Success criteria for diagnosis

1. **VC:** Either (a) stable trigger + hga=0 in ≥3/5 seeds, or (b) documented “never triggers” with low hidden_std — publish as failure mode.
2. **Vigilant:** Report mean ± std visible @ s99; if std > 0.05, lead post with instability.
3. Figures 1–2 regenerated including new seeds when Wave 1 completes.

## Launch

```bash
bash scripts/launch_sweep.sh wave1   # may hit 10-queue cap
python scripts/poll_sweeps.py        # fetch completed + refresh figures
```

`vig-p3-seed-*.toml` launched; IDs in `analysis/sweep_runs.json`.
