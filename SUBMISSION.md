# Reward Hacking Sprint — Submission Package

**Researcher:** austindixson  
**Track:** Reward Hacking (Prime Intellect Sprints, May 2026)  
**Title:** From Gradient Circuit Breakers to Behavioral Suppression: When Variance-Based Interventions Stop Reward Hacking

---

## One-Line Summary

A vigilance layer on `backdoor-ifeval` monitors within-batch hidden reward variance and kills the **hidden reward gradient** before exploitation consolidates — improving visible task performance by up to +48% on 1B models. **Full behavioral elimination** requires post-trigger penalties or visible constraints; see [METRICS.md](METRICS.md).

---

## Core Contribution (Sprint Prompt: Hacking Detection)

**Question:** Can you predict reward hacking from reward distributions in the first ~20 steps and intervene?

**Answer:** Yes for **gradient removal**. Cumulative group-level hidden variance triggers at steps 8–12 (before hidden behavior >10% on silver), sets effective `hidden_weight = 0`, and redirects gradient to the visible IFEval task. Behavioral rates can persist on high-baseline keywords (e.g. practice) even after the gradient is zero.

---

## Hub Environment

| ID | Version | Link |
|----|---------|------|
| `austindixson/backdoor-ifeval-vigilant` | 0.2.0 | https://app.primeintellect.ai/dashboard/environments/austindixson/backdoor-ifeval-vigilant |

Install and run:

```bash
prime env install austindixson/backdoor-ifeval-vigilant
prime train run --yes configs/vigilant-early-warning.toml
prime train run --yes configs/sprint-vigilant.toml   # FREE sprint model
```

**v0.2.0 additions:** `hidden_reward_mode` (binary/continuous/multi/sycophancy), `trigger_mode` (variance/oracle/random), `intervention_mode` (kill_gradient/behavior_penalty/visible_constraint).

---

## Key Training Runs (Reproduce)

| Experiment | Run ID | Config |
|------------|--------|--------|
| Vigilant th=5 (1B, agg-all) | `vd3qru13mag872bzd45t5dkc` | `configs/vigilant-early-warning.toml` |
| Control (1B, agg-all) | `k78uzf6leoyjqa543kcdjwbu` | `configs/vigilant-control.toml` |
| Sprint FREE vigilant | `tyd30ujks9tw9xjtcmfvgc32` | `configs/sprint-vigilant.toml` |
| Llama 3B sprint control | `k1jaocjlrfcu5tc2jgp2e2jx` | `configs/llama3b-sprint-control.toml` |
| Llama 3B sprint vigilant | `s1tyeiz6ve5a986wqzogpg0t` | `configs/llama3b-sprint-vigilant.toml` |
| Ablation: no-hidden 1B | `o682abhxdzw48upajgy674pr` | `configs/ablation-1b-no-hidden.toml` |
| Ablation: oracle 1B | `e56bi4yzmhn1i10pnm819kgj` | `configs/ablation-1b-oracle.toml` |

Full ablation matrix: [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md)

```bash
prime train metrics vd3qru13mag872bzd45t5dkc --plain --min-step 99 --max-step 99
prime train metrics k78uzf6leoyjqa543kcdjwbu --plain --min-step 99 --max-step 99
```

---

## Headline Results (1B, meta-llama, agg-all, hw=0.5)

| Run | s99 Visible | s99 Hidden (behavior) | Gradient killed? |
|-----|-------------|----------------------|------------------|
| Control | 0.488 | 0.167 | — |
| Vigilant th=5 | **0.717** | **0.000** | yes @ s12 |

Sprint model (`sprints/Llama-3.2-1B`, diff=3, average): visible **0.998**, hidden **0.040**, trigger s10, **$0.00**.

3B sprint: control s99 hid=**1.0**; vigilant s99 vis=**0.908**, hid=**0.0** (gradient removed).

---

## Strict elimination criteria (not yet met globally)

1. Hidden gradient zero after trigger ✓ (vigilant runs)  
2. Hidden **behavior** &lt;5% at s99 ✗ on practice (0.579) and some health runs  
3. Visible ≥ control ✓  
4. Two-seed replication — partial  
5. Beats random/oracle ablations — **in progress** ([EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md))  
6. Non-keyword hack mitigated — **in progress** (continuous/multi/sycophancy configs)

---

## Artifacts for Reviewers

| File | Role |
|------|------|
| [article.md](article.md) / [index.html](index.html) | Public narrative |
| [README.md](README.md) | Overview + results tables |
| [SPRINT_REPORT.md](SPRINT_REPORT.md) | Technical appendix |
| [METRICS.md](METRICS.md) | Gradient vs behavior definitions |
| [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) | Pre-registered ablations |
| [configs/](configs/) | All training TOMLs |
| [environments/backdoor_ifeval_vigilant/](environments/backdoor_ifeval_vigilant/) | Environment source |

---

## Where to Submit

Post this package to the May 2026 Reward Hacking sprint channel. Include:

1. Environment link: `austindixson/backdoor-ifeval-vigilant` v0.2.0  
2. This file or README link  
3. Narrative: `index.html`  
4. Key run IDs above  

---

## References

- [Prime Intellect: Systematic Reward Hacking](https://primeintellect.ai/blog/reward-hacking)
- Base env: `prime/backdoor-ifeval-all`
