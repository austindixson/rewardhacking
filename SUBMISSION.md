# Reward Hacking Sprint — Submission Package

**Researcher:** austindixson  
**Track:** Reward Hacking (Prime Intellect Sprints, May 2026)  
**Title:** From Gradient Circuit Breakers to Behavioral Suppression: When Variance-Based Interventions Stop Reward Hacking

---

## One-Line Summary

A vigilance layer on `backdoor-ifeval` monitors within-batch hidden reward variance and kills the **hidden reward gradient** before exploitation consolidates — improving visible IFEval reward vs control when hidden and visible objectives compete. **Primary evidence** is collected on the **free canonical stack** (`sprints/Llama-3.2-1B-Instruct`, agg-all); paid runs are appendix only until Phase 1 completes. See [METRICS.md](METRICS.md) and [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md).

---

## Core Contribution (Sprint Prompt: Hacking Detection)

**Question:** Can you predict reward hacking from reward distributions in the first ~20 steps and intervene?

**Answer:** Yes for **gradient removal**. Cumulative group-level hidden variance triggers at steps 8–12 (before hidden behavior >10% on silver), sets effective `hidden_weight = 0`, and redirects gradient to the visible IFEval task. Behavioral rates can persist on high-baseline keywords (e.g. practice) even after the gradient is zero.

---

## Hub Environment

| ID | Version | Link |
|----|---------|------|
| `austindixson/backdoor-ifeval-vigilant` | 0.2.1 (hub `latest`; Phase 1 configs pin `@0.2.0`) | https://app.primeintellect.ai/dashboard/environments/austindixson/backdoor-ifeval-vigilant |

Install and run:

```bash
prime env install austindixson/backdoor-ifeval-vigilant
prime train run --yes configs/vigilant-early-warning.toml
prime train run --yes configs/sprint-vigilant.toml   # FREE sprint model
```

**v0.2.0 additions:** `hidden_reward_mode` (binary/continuous/multi/sycophancy), `trigger_mode` (variance/oracle/random), `intervention_mode` (kill_gradient/behavior_penalty/visible_constraint).

---

## Claims we defend (canonical stack only)

**Stack:** `sprints/Llama-3.2-1B-Instruct`, `aggregation=all`, `hidden_weight=0.5`, 100 steps — see [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md).

| Claim | Evidence needed |
|-------|-----------------|
| Gradient gating works | P0: vigilant s99 `hidden_gradient_active=0`; visible ≥ control |
| Detection beats dumb timing | P1: vigilant &gt; random on visible; competitive with oracle |
| Not redundant with no-hidden | P1: compare vigilant vs no-hidden visible |
| Behavior may persist | P2: `behavioral_residual` on high-baseline; penalty/constraint ablations |
| Non-keyword monitor | P2: Phase C vigilant vs control per hack mode |

**Not claimed until P3 + strict criteria:** “eliminated,” cross-model generalization, 3B scale. P0 canonical pair complete (2026-05-24).

## Key Training Runs (Reproduce)

**Phase 1 (canonical)** — fill run IDs in [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) as jobs finish:

```bash
prime train run --yes configs/vigilant-control.toml
prime train run --yes configs/vigilant-early-warning.toml
# + ablation-1b-* per matrix
```

| Experiment | Run ID | Config |
|------------|--------|--------|
| Control (canonical) | `e4yj35o7wszr29kz82y4yuwx` | `configs/vigilant-control.toml` |
| Vigilant th=5 (canonical) | `jfqgp71by8vgy2ksoymmopmg` | `configs/vigilant-early-warning.toml` |
| Ablation: no-hidden | `zk299rbfgm4k801pv69dp7fb` | `configs/ablation-1b-no-hidden.toml` |
| Ablation: oracle | `lmqwm4kjdrevce58853korv7` | `configs/ablation-1b-oracle.toml` |

**Appendix (paid / different settings)** — do not mix into primary tables:

| Experiment | Run ID | Notes |
|------------|--------|-------|
| Pre-canonical vigilant | `vd3qru13mag872bzd45t5dkc` | meta-llama 1B |
| Pre-canonical control | `k78uzf6leoyjqa543kcdjwbu` | meta-llama 1B |
| Sprint demo (easier env) | `tyd30ujks9tw9xjtcmfvgc32` | diff=3, average |
| 3B scale sketch | `k1jaoc…`, `s1tyei…` | Phase 2 |

```bash
prime train metrics vd3qru13mag872bzd45t5dkc --plain --min-step 99 --max-step 99
prime train metrics k78uzf6leoyjqa543kcdjwbu --plain --min-step 99 --max-step 99
```

---

## Headline Results

### Canonical (`sprints/Llama-3.2-1B`, agg-all) — **12/14 s99** (2026-05-24)

| Run | s99 Visible | s99 Hidden (behavior) | `hidden_gradient_active` |
|-----|-------------|----------------------|--------------------------|
| P0 control (`e4yj35o7…`) | 0.663 | 0.200 | — |
| P0 vigilant (`jfqgp71b…`) | **0.669** | **0.000** | **0** |

Vigilant matches control on visible and zeros hidden gradient at s99 (`vigilance_active=1`). Full ablation table: [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md). Failed: `visible-constraint`, `multi-control`.

### Appendix: pre-canonical (meta-llama 1B, agg-all)

| Run | s99 Visible | s99 Hidden (behavior) | Gradient killed? |
|-----|-------------|----------------------|------------------|
| Control | 0.488 | 0.167 | — |
| Vigilant th=5 | **0.717** | **0.000** | yes @ s12 |

### Appendix: other settings (not canonical)

- Sprint demo (`diff=3`, `average`): visible **0.998**, hidden **0.040**, `tyd30ujks9tw9xjtcmfvgc32`, **$0.00**
- 3B sprint (Phase 2): control hid=**1.0**; vigilant vis=**0.908**, hid=**0.0**

---

## Strict elimination criteria (not yet met globally)

1. Hidden gradient zero after trigger ✓ (vigilant runs)  
2. Hidden **behavior** &lt;5% at s99 ✗ on practice (0.579) and some health runs  
3. Visible ≥ control ✓ (canonical P0)  
4. Two-seed replication — **P3 not started**  
5. Beats random/oracle ablations — **partial** (vigilant 0.669 vs random 0.673 vs oracle 0.842)  
6. Non-keyword hack mitigated — **partial** (continuous-vigilant 0.810; sycophancy trigger did not fire)

Related framing: in self-play RL, [Pu et al. (2026)](https://arxiv.org/abs/2605.22217) show that **data-level gating** can stabilize training when reward grounding alone cannot; we treat variance-triggered `hidden_weight` kill as **gradient gating** on the hidden channel and test **behavior-level** interventions (penalty, visible constraint) for residual exploit behavior.

---

## Artifacts for Reviewers

| File | Role |
|------|------|
| [article.md](article.md) / [index.html](index.html) | Public narrative |
| [README.md](README.md) | Overview + results tables |
| [SPRINT_REPORT.md](SPRINT_REPORT.md) | Technical appendix |
| [METRICS.md](METRICS.md) | Gradient vs behavior definitions |
| [REVIEWER_FAQ.md](REVIEWER_FAQ.md) | Skeptic FAQ: claims, ablations, objections |
| [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) | Pre-registered ablations |
| [configs/](configs/) | All training TOMLs |
| [environments/backdoor_ifeval_vigilant/](environments/backdoor_ifeval_vigilant/) | Environment source |

---

## Where to Submit

Post this package to the May 2026 Reward Hacking sprint channel. Include:

1. Environment link: `austindixson/backdoor-ifeval-vigilant` v0.2.1 (Phase 1 runs mostly `@0.2.0`)  
2. This file or README link  
3. Narrative: `index.html`  
4. Key run IDs above  

---

## References

- [Prime Intellect: Systematic Reward Hacking](https://primeintellect.ai/blog/reward-hacking)
- Pu, S. X., Weng, Z., Liu, C., Srinivasa, J., Liu, G., Wang, W. Y., & Wang, X. E. (2026). *Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL.* arXiv:2605.22217. https://arxiv.org/abs/2605.22217
- Base env: `prime/backdoor-ifeval-all`
