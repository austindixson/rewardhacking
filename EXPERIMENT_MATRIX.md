# Pre-Registered Experiment Matrix

**Env:** `austindixson/backdoor-ifeval-vigilant` — Hub **v0.2.3** (visible-constraint tuple fix); Phase 1 pins `@0.2.0` unless noted  
**Metric definitions:** [METRICS.md](METRICS.md) · **Reviewer FAQ:** [REVIEWER_FAQ.md](REVIEWER_FAQ.md)

## Takeaways (2026-05-24)

**On track for a defensible sprint submission; not on track for “breakthrough / elimination.”**

| Finding | Evidence | Implication |
|---------|----------|-------------|
| **Gradient gating works (binary, canonical)** | P0 + P3 vigilant: `hidden_gradient_active=0`, `vigilance_active=1` | Core claim: variance spike → kill hidden gradient |
| **Visible gain is real but noisy** | P0 vis **0.669** ≥ control **0.663**; P3 vis **0.591** (44% `zero_advantage`) | Report **range**, not single s99 |
| **Detector ≈ random on visible (binary)** | vigilant **0.669** vs random **0.673** vs oracle **0.842** | Timing signal does **not** beat dumb schedule on visible |
| **Behavior levers beat variance kill (visible)** | behavior-penalty **0.887** | Post-trigger shaping ≠ gradient kill |
| **Non-keyword: monitor often silent** | continuous/sycophancy vigilant: **no variance trigger** | Variance detector is **regime-specific** (binary agg-all) |
| **2B continuous: random ≈ vigilant** | control **0.783**, vigilant **0.810**, random **0.788** (hga=0) | Hold-out does **not** show vigilant &gt; random on s99 |
| **visible-constraint** | v0.2.3 run: no trigger, hga=1 | **Drop** from claims; needs separate debug |
| **Control replicate unstable** | P3 control rep vis **0.933**, hid **0** (75% filtered) | Do **not** use for replication narrative |

**Ship claim:** *Variance-triggered gradient gating on hidden reward can preserve or improve visible IFEval on the keyword backdoor while zeroing the hidden gradient — with limits on replication variance, ablation timing, and non-keyword hacks.*

---

## Roadmap (phases)

| Phase | Name | Status |
|-------|------|--------|
| **1** | Canonical 1B + ablations + P3 | **CLOSED** — 14/14 terminal outcomes |
| **2** | Breakthrough + seed sweeps | **2B done**; Wave 1 VC/vigilant seeds **in flight** (see `analysis/sweep_runs.json`) |
| — | Appendix paid runs | Historical |

---

## Canonical stack (Phase 1 primary claims)

| Field | Value |
|-------|--------|
| Model | `sprints/Llama-3.2-1B-Instruct` |
| Env | `aggregation = "all"`, `hidden_weight = 0.5` |
| Training | 100 steps, batch 128, 8 rollouts/example |
| Vigilance | `spike_threshold = 5` where applicable |

### Phase 1 progress (last updated 2026-05-24)

| Bucket | s99 | Failed / inconclusive |
|--------|-----|------------------------|
| P0 headline | 2/2 | — |
| Phase A | 4/5 | visible-constraint (×3 inconclusive) |
| Phase C | 6/7 | multi-control (×2) |
| P3 vigilant + control rep | 2/2 | control rep anomalous |
| **Total** | **14/14** outcomes | 2 infra failures; constraint dropped |

**Strict elimination (1–6):** partial only — see [Success criteria](#success-criteria).

---

## Success criteria

### Phase 1 — strict “elimination”

1. `hidden_gradient_active` = 0 after trigger — **✓** (binary vigilant runs)  
2. Behavioral &lt;5% at s99 — **✓** on P0/P3 vigilant s99  
3. Visible ≥ control — **✓** P0; **✗** P3 vigilant alone  
4. Two hosted vigilant runs — **✓** mechanism; **partial** visible band  
5. Beats random/oracle on visible — **✗** (≈ random; &lt; oracle)  
6. Non-keyword mitigated — **✗** / inconclusive  

### Phase 2 breakthrough bar

| ID | Status (2026-05-24) |
|----|---------------------|
| **2A** | Not started (analyze timelines on P1 runs) |
| **2B** | Continuous quartet **done** — oracle s99 vis **0.392** (below random **0.788**) |
| **Sweeps** | Wave 1: 8 VC + 3 vig P0 queued/running; P3 vig blocked at 10-queue cap |
| **2C** | Blocked (visible-constraint inconclusive) |
| **2D** | Deferred (paid 1B/3B) |

---

## Phase 1 — Canonical 1B

### P0 — Headline pair

| Config | Run ID | s99 vis | s99 hid | s99 hga | Notes |
|--------|--------|---------|---------|---------|-------|
| `vigilant-control.toml` | `e4yj35o7wszr29kz82y4yuwx` | 0.663 | 0.200 | — | |
| `vigilant-early-warning.toml` | `jfqgp71by8vgy2ksoymmopmg` | 0.669 | 0.000 | 0.00 | `vigilance_active=1` |

### Phase A

| Config | Run ID | s99 vis | s99 hid | Notes |
|--------|--------|---------|---------|-------|
| `ablation-1b-no-hidden.toml` | `zk299rbfgm4k801pv69dp7fb` | 0.692 | 0.000 | |
| `ablation-1b-oracle.toml` | `lmqwm4kjdrevce58853korv7` | 0.842 | 0.000 | |
| `ablation-1b-random.toml` | `dt0i5dzt479xpo7c9ibq9lry` | 0.673 | 0.000 | |
| `ablation-1b-behavior-penalty.toml` | `vn591wsn598b4n1bnunxkld4` | 0.887 | 0.000 | |
| `ablation-1b-visible-constraint.toml` | `gzis11zm0y3egrtf9rrtlb8s` | 0.975 | 0.000 | **INCONCLUSIVE** — `vigilance_active=0`, hga=1; prior runs failed |

### Phase C

| Config | Run ID | s99 vis | s99 hid | Notes |
|--------|--------|---------|---------|-------|
| `ablation-1b-continuous-control.toml` | `vjeuarzrms4tjag9ywid5p2x` | 0.783 | 0.000 | |
| `ablation-1b-continuous-vigilant.toml` | `g0va3w9ixj3xnw8frd1ckkgs` | 0.810 | 0.000 | no trigger |
| `ablation-1b-multi-control.toml` | `esg5nupga…`, `nm01t4jk…` | — | — | **FAILED** |
| `ablation-1b-multi-vigilant.toml` | `bk5vvkvw2txpinh78yja1re1` | 0.646 | 0.000 | |
| `ablation-1b-sycophancy-control.toml` | `qvzpldz61ykv34srhgget587` | 0.727 | 0.000 | |
| `ablation-1b-sycophancy-vigilant.toml` | `h16dbek6i9142rjwa4ii9r31` | 0.910 | 0.000 | no trigger; hga=1 |
| `ablation-1b-sycophancy-penalty.toml` | `lhwlyyk4xvhtrcpfiowmw269` | 0.700 | 0.000 | |

### P3 — Replication

| Config | Run ID | s99 vis | s99 hid | s99 hga | Notes |
|--------|--------|---------|---------|---------|-------|
| `p3-vigilant-replicate.toml` | `q7lktv5shrn18el0t4wi2vwq` | 0.591 | 0.000 | 0.00 | mechanism ✓ |
| `p3-control-replicate.toml` | `n2ebo5pxrok9f87rpeazbi9e` | 0.933 | 0.000 | — | **anomalous** — 75% `zero_advantage`; not comparable to P0 control |

---

## Phase 2 — Breakthrough program

### 2B — Continuous hold-out (preregistered triple + oracle)

| Method | Config | Run ID | s99 vis | s99 hid | hga | Notes |
|--------|--------|--------|---------|---------|-----|-------|
| control | `ablation-1b-continuous-control.toml` | `vjeuarzr…` | 0.783 | 0.000 | — | Phase 1 |
| vigilant | `ablation-1b-continuous-vigilant.toml` | `g0va3w9i…` | 0.810 | 0.000 | on* | no variance trigger |
| random @ g5 | `phase2b-continuous-random.toml` | `k9m87rxtcd2ukk6fbx9bgv4y` | 0.788 | 0.000 | 0.00 | `vigilance_active=1` |
| oracle @ g5 | `phase2b-continuous-oracle.toml` | `o24i8bnnd2emcsoddu5mhrvb` | **0.392** | 0.000 | 0.00 | early kill ~step 20; 44% `zero_advantage` |

\*continuous-vigilant s99 `hidden_gradient_active` was 0.56 in Phase 1 read; random run hga=0.

**2B s99 read:** **random (0.788) ≫ oracle (0.392)** on visible despite both killing gradient — early oracle kill may hurt visible on continuous. Variance vigilant (0.810) best s99 but often **no trigger**. Strong **negative** for “oracle = ceiling” on this geometry.

### Wave 1 seed sweeps (diagnosis)

Track live: `analysis/sweep_runs.json` · poll: `python scripts/poll_sweeps.py`

| Arm | Status |
|-----|--------|
| `vc-seed-01` … `05` + variants | RUNNING/QUEUED (`f5b7q961…`, `vix2ld2s…`, …) |
| `vig-p0-seed-01` … `03` | QUEUED |
| `vig-p3-seed-*` | **Blocked** — Prime 10-queue cap; launch when queue drains |

### 2A / 2C / 2D

See prior prereg factors in git history; **2A** = regret timelines on binary P1 grid; **2C** after constraint debug; **2D** last.

---

## Appendix: pre-canonical runs

| Config | Run ID | s99 vis | s99 hid |
|--------|--------|---------|---------|
| control (meta-llama 1B) | `k78uzf6leoyjqa543kcdjwbu` | 0.488 | 0.167 |
| vigilant (meta-llama 1B) | `vd3qru13mag872bzd45t5dkc` | 0.717 | 0.000 |

---

## Commands

```bash
prime train list --mine --plain --num 10
prime train metrics <RUN_ID> --plain --min-step 99 --max-step 99
prime train metrics <RUN_ID> --plain --min-step 0 --max-step 99   # regret timelines
prime train run --yes configs/phase2b-continuous-oracle.toml
```
