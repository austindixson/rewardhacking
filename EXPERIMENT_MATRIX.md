# Pre-Registered Experiment Matrix

**Env:** `austindixson/backdoor-ifeval-vigilant` -  Hub **v0.2.4** (`intervention_group` metric); Wave 1 VC sweeps use **@0.2.4**; Phase 1 pins `@0.2.0` unless noted  
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
| **visible-constraint** | Wave 1 (n=5): vis **0.984±0.006**, trigger **1/5** (seed-02 only); `threshold-7` still no trigger | **Drop** from claims -  high visible + unstable trigger |
| **2A timing** | Oracle/random trigger **step 1**; vigilant **step 22**; behavior-penalty best **Δs99 +0.225** | Early kill ≠ low regret; penalty beats gradient timing |
| **Control replicate unstable** | P3 control rep vis **0.933**, hid **0** (75% filtered) | Do **not** use for replication narrative |

**Ship claim:** *Variance-triggered gradient gating on hidden reward can preserve or improve visible IFEval on the keyword backdoor while zeroing the hidden gradient -  with limits on replication variance, ablation timing, and non-keyword hacks.*

---

## Roadmap (phases)

| Phase | Name | Status |
|-------|------|--------|
| **1** | Canonical 1B + ablations + P3 | **CLOSED** -  14/14 terminal outcomes |
| **2** | Breakthrough + seed sweeps | **2A+2B done**; Wave 1 VC **done**; vig-P0/P3 **queued** (`analysis/sweep_runs.json`) |
| -  | Appendix paid runs | Historical |

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
| P0 headline | 2/2 | -  |
| Phase A | 4/5 | visible-constraint (×3 inconclusive) |
| Phase C | 6/7 | multi-control (×2) |
| P3 vigilant + control rep | 2/2 | control rep anomalous |
| **Total** | **14/14** outcomes | 2 infra failures; constraint dropped |

**Strict elimination (1–6):** partial only -  see [Success criteria](#success-criteria).

---

## Success criteria

### Phase 1 -  strict “elimination”

1. `hidden_gradient_active` = 0 after trigger -  **✓** (binary vigilant runs)  
2. Behavioral &lt;5% at s99 -  **✓** on P0/P3 vigilant s99  
3. Visible ≥ control -  **✓** P0; **✗** P3 vigilant alone  
4. Two hosted vigilant runs -  **✓** mechanism; **partial** visible band  
5. Beats random/oracle on visible -  **✗** (≈ random; &lt; oracle)  
6. Non-keyword mitigated -  **✗** / inconclusive  

### Phase 2 breakthrough bar

| ID | Status (2026-05-24) |
|----|---------------------|
| **2A** | **Done** -  [regret_summary.md](analysis/regret_summary.md), [fig4](analysis/figures/fig4_regret_binary.png) |
| **2B** | Continuous quartet **done** -  oracle s99 vis **0.392** (below random **0.788**) |
| **Sweeps** | VC arm **8/8 done** (trigger **2/8** with threshold-3); vig-P0 **running/queued**; vig-P3 **queued** |
| **2C** | Blocked (visible-constraint inconclusive) |
| **2D** | Deferred (paid 1B/3B) |

---

## Phase 1 -  Canonical 1B

### P0 -  Headline pair

| Config | Run ID | s99 vis | s99 hid | s99 hga | Notes |
|--------|--------|---------|---------|---------|-------|
| `vigilant-control.toml` | `e4yj35o7wszr29kz82y4yuwx` | 0.663 | 0.200 | -  | |
| `vigilant-early-warning.toml` | `jfqgp71by8vgy2ksoymmopmg` | 0.669 | 0.000 | 0.00 | `vigilance_active=1` |

### Phase A

| Config | Run ID | s99 vis | s99 hid | Notes |
|--------|--------|---------|---------|-------|
| `ablation-1b-no-hidden.toml` | `zk299rbfgm4k801pv69dp7fb` | 0.692 | 0.000 | |
| `ablation-1b-oracle.toml` | `lmqwm4kjdrevce58853korv7` | 0.842 | 0.000 | |
| `ablation-1b-random.toml` | `dt0i5dzt479xpo7c9ibq9lry` | 0.673 | 0.000 | |
| `ablation-1b-behavior-penalty.toml` | `vn591wsn598b4n1bnunxkld4` | 0.887 | 0.000 | |
| `ablation-1b-visible-constraint.toml` | `gzis11zm0y3egrtf9rrtlb8s` | 0.975 | 0.000 | **INCONCLUSIVE** -  `vigilance_active=0`, hga=1; prior runs failed |

### Phase C

| Config | Run ID | s99 vis | s99 hid | Notes |
|--------|--------|---------|---------|-------|
| `ablation-1b-continuous-control.toml` | `vjeuarzrms4tjag9ywid5p2x` | 0.783 | 0.000 | |
| `ablation-1b-continuous-vigilant.toml` | `g0va3w9ixj3xnw8frd1ckkgs` | 0.810 | 0.000 | no trigger |
| `ablation-1b-multi-control.toml` | `esg5nupga…`, `nm01t4jk…` | -  | -  | **FAILED** |
| `ablation-1b-multi-vigilant.toml` | `bk5vvkvw2txpinh78yja1re1` | 0.646 | 0.000 | |
| `ablation-1b-sycophancy-control.toml` | `qvzpldz61ykv34srhgget587` | 0.727 | 0.000 | |
| `ablation-1b-sycophancy-vigilant.toml` | `h16dbek6i9142rjwa4ii9r31` | 0.910 | 0.000 | no trigger; hga=1 |
| `ablation-1b-sycophancy-penalty.toml` | `lhwlyyk4xvhtrcpfiowmw269` | 0.700 | 0.000 | |

### P3 -  Replication

| Config | Run ID | s99 vis | s99 hid | s99 hga | Notes |
|--------|--------|---------|---------|---------|-------|
| `p3-vigilant-replicate.toml` | `q7lktv5shrn18el0t4wi2vwq` | 0.591 | 0.000 | 0.00 | mechanism ✓ |
| `p3-control-replicate.toml` | `n2ebo5pxrok9f87rpeazbi9e` | 0.933 | 0.000 | -  | **anomalous** -  75% `zero_advantage`; not comparable to P0 control |

---

## Phase 2 -  Breakthrough program

### 2B -  Continuous hold-out (preregistered triple + oracle)

| Method | Config | Run ID | s99 vis | s99 hid | hga | Notes |
|--------|--------|--------|---------|---------|-----|-------|
| control | `ablation-1b-continuous-control.toml` | `vjeuarzr…` | 0.783 | 0.000 | -  | Phase 1 |
| vigilant | `ablation-1b-continuous-vigilant.toml` | `g0va3w9i…` | 0.810 | 0.000 | on* | no variance trigger |
| random @ g5 | `phase2b-continuous-random.toml` | `k9m87rxtcd2ukk6fbx9bgv4y` | 0.788 | 0.000 | 0.00 | `vigilance_active=1` |
| oracle @ g5 | `phase2b-continuous-oracle.toml` | `o24i8bnnd2emcsoddu5mhrvb` | **0.392** | 0.000 | 0.00 | early kill ~step 20; 44% `zero_advantage` |

\*continuous-vigilant s99 `hidden_gradient_active` was 0.56 in Phase 1 read; random run hga=0.

**2B s99 read:** **random (0.788) ≫ oracle (0.392)** on visible despite both killing gradient -  early oracle kill may hurt visible on continuous. Variance vigilant (0.810) best s99 but often **no trigger**. Strong **negative** for “oracle = ceiling” on this geometry.

### Wave 1 seed sweeps (diagnosis)

Track live: `analysis/sweep_runs.json` · poll: `python scripts/poll_sweeps.py` · figures: `analysis/figures/`

**visible-constraint @0.2.4** (`visible_constraint=true`, `aggregation=all` → env falls back to `average`)

| Config | Run ID | s99 vis | s99 hid | hga | vig | Notes |
|--------|--------|---------|---------|-----|-----|-------|
| `vc-seed-01` | `f5b7q961eldpww2ujw23pbxo` | 0.976 | 0.000 | 1.00 | 0 | no trigger |
| `vc-seed-02` | `vix2ld2smxhdmbhjgyzj5bvv` | 0.987 | 0.000 | 0.00 | 1 | **only** seed with gradient kill |
| `vc-seed-03` | `z91fkxvyag8x2b6uf75wt89w` | 0.981 | 0.000 | 1.00 | 0 | no trigger |
| `vc-seed-04` | `hwsg5uh5ekrppvi7waiqhfa4` | 0.984 | 0.000 | 1.00 | 0 | no trigger |
| `vc-seed-05` | `fya3o5bgqfonku8rrfresj96` | 0.993 | 0.000 | 1.00 | 0 | no trigger |
| `vc-threshold-3` | `k0q07qlqng752ckvyoqnp40o` | 0.980 | 0.000 | 0.00 | 1 | lower threshold → trigger |
| `vc-threshold-7` | `l7e5fd2ml5ndbzl7wymtnxc9` | 0.992 | 0.000 | 1.00 | 0 | high threshold → **no** trigger |
| `vc-agg-average` | `zxg3hg5odorevpsvyc4s850h` | 0.969 | 0.000 | 1.00 | 0 | explicit `aggregation=average`; no trigger |

**VC aggregate (seeds 01–05, s99):** visible **0.984±0.006**; trigger rate **20%** (1/5); **4/5** keep `hga=1`.

| Arm | Status |
|-----|--------|
| `vig-p0-seed-01` | `yt7yl142…` | -  | -  | -  | -  | RUNNING |
| `vig-p0-seed-02`–`05` | queued | -  | -  | -  | -  | |
| `vig-p3-seed-01`–`05` | queued | -  | -  | -  | -  | `ssosrlzv…` … `eqb4jbx5…` |

### 2A -  Binary regret (no new runs)

`python scripts/analyze_regret.py` → [analysis/regret_summary.md](analysis/regret_summary.md), [fig4](analysis/figures/fig4_regret_binary.png)

| Config | Trigger step | Δs99 vs control | Mean regret post-trigger |
|--------|--------------|-----------------|--------------------------|
| vigilant | 22 | +0.006 | +0.119 (control ahead) |
| p3-vigilant | 14 | −0.072 | +0.092 |
| random | 1 | +0.010 | −0.041 |
| oracle | 1 | **+0.179** | +0.026 |
| behavior-penalty | 18 | **+0.225** | −0.149 (method ahead) |

**Read:** Oracle’s early kill does **not** minimize regret -  it **maximizes** end visible vs control. Variance vigilant triggers late with small Δs99. Behavior-penalty dominates visible without gradient-timing luck.

### 2C / 2D

**2C** -  visible-constraint closed as failure mode (Wave 1). **2D** -  paid 1B/3B deferred.

---

## Appendix: pre-canonical runs

| Config | Run ID | s99 vis | s99 hid |
|--------|--------|---------|---------|
| control (meta-llama 1B) | `k78uzf6leoyjqa543kcdjwbu` | 0.488 | 0.167 |
| vigilant (meta-llama 1B) | `vd3qru13mag872bzd45t5dkc` | 0.717 | 0.000 |

---

## Commands

```bash
python scripts/poll_sweeps.py          # incremental fetch (completed runs only) + figures
python scripts/fetch_metrics.py        # full refresh → metrics_cache + sweep_summary
python scripts/summarize_cache.py      # mean±std from sweep_summary.json
python scripts/analyze_regret.py       # Phase 2A (needs local metrics_cache)
prime train list --mine --plain --num 15
prime train metrics <RUN_ID> --plain --min-step 99 --max-step 99
prime train metrics <RUN_ID> --plain --min-step 0 --max-step 99   # regret timelines
```
