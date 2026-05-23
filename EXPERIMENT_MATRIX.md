# Pre-Registered Experiment Matrix

**Env:** `austindixson/backdoor-ifeval-vigilant` v0.2.0+  
**Metric definitions:** [METRICS.md](METRICS.md) · **Reviewer FAQ:** [REVIEWER_FAQ.md](REVIEWER_FAQ.md)

## Canonical stack (all primary claims)

Until this phase is complete, **do not cite paid runs as primary evidence.**

| Field | Value |
|-------|--------|
| Model | `sprints/Llama-3.2-1B-Instruct` (free sprint tier) |
| Env | `aggregation = "all"`, `hidden_weight = 0.5` |
| Training | 100 steps, batch 128, 8 rollouts/example |
| Vigilance | `spike_threshold = 5` where applicable |

**Paid models** (`meta-llama/*`, 3B, Qwen) → [Appendix: exploratory runs](#appendix-exploratory-paid-runs) only, after the canonical matrix is filled.

### Primary contrasts (fill s99 on canonical stack)

Report for every row: `visible_reward`, `hidden_reward`, `hidden_gradient_active`, `behavioral_residual`, trigger step.

| Priority | Contrast | Configs | Question |
|----------|----------|---------|----------|
| **P0** | Method vs baseline | `vigilant-early-warning` vs `vigilant-control` | Does variance + gradient kill improve visible? |
| **P1** | Detection vs timing | vigilant vs `ablation-1b-oracle` vs `ablation-1b-random` | Does the signal beat scheduled/random kill? |
| **P1** | vs no backdoor | vigilant vs `ablation-1b-no-hidden` | Is vigilance better than never training hidden? |
| **P2** | Behavior levers | `behavior-penalty`, `visible-constraint` vs vigilant | Does post-trigger behavior drop (&lt;5%) on high-baseline? |
| **P2** | Non-keyword | Phase C control/vigilant pairs | Does std spike before continuous/multi/sycophancy exploit? |
| **P3** | Replication | Second run of P0 vigilant (and optionally control) | Same conclusion on repeat hosted run |

**Submission rule:** Judges see only canonical-stack tables + one honest limitations paragraph. Paid appendix is optional “future work / scale sketch.”

### Phase 1 progress (last updated 2026-05-23)

| Bucket | Done (s99) | In flight | Queued | Not launched |
|--------|------------|-----------|--------|--------------|
| P0 headline | 0 / 2 | 0 | **2** | — *(queued 2026-05-23)* |
| Phase A | 3 / 5 | 1–2 | 0 | — |
| Phase C | 1 / 7 | 0 | 6 | — |
| **Total** | **4 / 14** | | | |

**Early P1 read (canonical sprint 1B):** oracle visible **0.842** &gt; random **0.673** &gt; no-hidden **0.692**; both oracle/random killed hidden gradient at s99. P0 variance vigilant + control **queued** (`e4yj35o7…`, `jfqgp71b…`) — detection vs timing TBD until those finish.

---

## Success criteria (strict “elimination”)

Use **eliminated** only when all hold at step 99 **on the canonical stack**:

1. `hidden_gradient_active` = 0 after trigger  
2. `behavioral_residual` (or `hidden_reward`) &lt; 5%  
3. `visible_reward` ≥ matched control  
4. Replicates across two seeds or repeated hosted runs  
5. Beats random/oracle ablations on visible − hidden tradeoff  
6. At least one non-keyword hack channel (continuous, multi, or sycophancy)

Until then: **gradient removed**, **suppressed**, or **mitigated**.

---

## Phase 1 — Canonical 1B (agg-all, free sprint model)

### P0 — Headline pair

| Config | Condition | Run ID | s99 vis | s99 hid | s99 residual |
|--------|-----------|--------|---------|---------|--------------|
| `vigilant-control.toml` | Vigilance off | `e4yj35o7wszr29kz82y4yuwx` | | | *(queued)* |
| `vigilant-early-warning.toml` | Variance th=5 | `jfqgp71by8vgy2ksoymmopmg` | | | *(queued)* |

### Phase A — Detection vs intervention

| Config | Condition | Success criterion | Run ID | s99 vis | s99 hid | s99 residual |
|--------|-----------|-------------------|--------|---------|---------|--------------|
| `ablation-1b-no-hidden.toml` | hidden_weight=0 | Upper bound visible | `zk299rbfgm4k801pv69dp7fb` | 0.692 | 0.000 | — |
| `ablation-1b-oracle.toml` | Oracle @ g5 | vs variance | `lmqwm4kjdrevce58853korv7` | 0.842 | 0.000 | 0.000 |
| `ablation-1b-random.toml` | Random @ g5 | vs variance | `dt0i5dzt479xpo7c9ibq9lry` | 0.673 | 0.000 | 0.000 |
| `ablation-1b-behavior-penalty.toml` | Penalty post-trigger | residual &lt;5% | `vn591wsn598b4n1bnunxkld4` | | | *(running or queued)* |
| `ablation-1b-visible-constraint.toml` | Forbid word post-trigger | residual &lt;5% | `f9is26bj21gy6jh6dxm50a0i` | | | *(running)* |

*Notes (Phase A s99):* Oracle and random both have `hidden_gradient_active=0` at s99 (`vigilance_trigger=0`; kill is schedule/random, not variance). No-hidden is the visible ceiling without hidden training.

*Note:* `vigilant-early-warning.toml` is the variance vigilant row for P0/P1 comparisons (same env args as historical `vd3qru13…` but on canonical model).

### Phase C — Non-keyword hacks

| Config | Hack type | Vigilance | Run ID | s99 vis | s99 hid |
|--------|-----------|-----------|--------|---------|---------|
| `ablation-1b-continuous-control.toml` | Token density | off | `vjeuarzrms4tjag9ywid5p2x` | | | |
| `ablation-1b-continuous-vigilant.toml` | Token density | on | `g0va3w9ixj3xnw8frd1ckkgs` | | | |
| `ablation-1b-multi-control.toml` | Multi-channel | off | `esg5nupga1scshls9il8ssa4` | | | |
| `ablation-1b-multi-vigilant.toml` | Multi-channel | on | `bk5vvkvw2txpinh78yja1re1` | | | |
| `ablation-1b-sycophancy-control.toml` | Agreement phrases | off | `qvzpldz61ykv34srhgget587` | | | |
| `ablation-1b-sycophancy-vigilant.toml` | Agreement phrases | on | `h16dbek6i9142rjwa4ii9r31` | 0.910 | 0.000 | *(no variance trigger; spike≈0.59)* |
| `ablation-1b-sycophancy-penalty.toml` | Sycophancy + penalty | on | `lhwlyyk4xvhtrcpfiowmw269` | | | |

### P3 — Replication (canonical)

| Config | Run ID | s99 vis | s99 hid | Notes |
|--------|--------|---------|---------|-------|
| `vigilant-early-warning.toml` (rep 2) | *(after P0 completes)* | | | Same TOML, second hosted run |
| `vigilant-control.toml` (rep 2) | *(optional)* | | | |

---

## Phase 2 — Paid scale (after Phase 1 complete)

Do not start until **P0–P2 s99 columns are filled** on canonical 1B.

| Track | Model | Setting | Purpose |
|-------|--------|---------|---------|
| 3B sprint | `meta-llama/Llama-3.2-3B-Instruct` | diff=3, average | Scale anecdote (existing `k1jaoc…` / `s1tyei…`) |
| 3B ablations | meta-llama 3B | same as Phase A | Match detection ablations at scale |
| 1B hard replicate | `meta-llama/Llama-3.2-1B-Instruct` | agg-all | Confirm free-tier results on paid weights |
| Qwen / larger | TBD | fix zero_advantage first | Generalization |

Stopped / pending 3B ablation IDs: `vczctdfn…`, `mdu8dzma…` — relaunch in Phase 2 only.

---

## Appendix: exploratory (paid) runs

**Not used for primary claims.** Kept for engineering history and Phase 2 planning.

| Config | Run ID | Model | s99 vis | s99 hid | Notes |
|--------|--------|-------|---------|---------|-------|
| `vigilant-control.toml` | `k78uzf6leoyjqa543kcdjwbu` | meta-llama 1B | 0.488 | 0.167 | Pre-canonical headline control |
| `vigilant-early-warning.toml` | `vd3qru13mag872bzd45t5dkc` | meta-llama 1B | 0.717 | 0.000 | Pre-canonical headline vigilant |
| `sprint-vigilant.toml` | `tyd30ujks9tw9xjtcmfvgc32` | sprint 1B | ~0.998 | ~0.04 | **Different env** (diff=3, average) — compliance demo only |
| `llama3b-sprint-control.toml` | `k1jaocjlrfcu5tc2jgp2e2jx` | meta-llama 3B | 0.667 | 1.000 | Phase 2 |
| `llama3b-sprint-vigilant.toml` | `s1tyeiz6ve5a986wqzogpg0t` | meta-llama 3B | 0.908 | 0.000 | Phase 2 |
| meta-llama 1B ablations (stopped) | `o682…`, `e56…`, etc. | meta-llama 1B | partial | partial | Superseded by sprint IDs in Phase 1 |

---

## Deferred (not in v0.2.0)

| Intervention | Status |
|--------------|--------|
| Training-buffer example replacement | Requires orchestrator support — motivated by [Pu et al. (2026)](https://arxiv.org/abs/2605.22217) |
| Dynamic Goldilocks + vigilance combined | Two envs; single-run config TBD |
| Auto-calibrated detector (EWMA/CUSUM) | Phase 2+ comparison vs variance baseline |

---

## Monitor & fill results

```bash
prime train list --mine --plain --num 15
prime train metrics <RUN_ID> --plain --min-step 99 --max-step 99
```

Launch canonical P0 (if not already queued):

```bash
prime train run --yes configs/vigilant-control.toml
prime train run --yes configs/vigilant-early-warning.toml
```
