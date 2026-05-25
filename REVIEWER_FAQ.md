# Reviewer FAQ

Quick answers for skeptics reviewing the Reward Hacking sprint submission.  
**Hub env:** [`austindixson/backdoor-ifeval-vigilant`](https://app.primeintellect.ai/dashboard/environments/austindixson/backdoor-ifeval-vigilant) v0.2.3 (`latest`; Phase 1 TOMLs pin `@0.2.0` unless noted)  
**Deeper tables:** [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) · **Metrics:** [METRICS.md](METRICS.md) · **Submission:** [SUBMISSION.md](SUBMISSION.md)

---

## What are you actually claiming?

**Claimed (supported today):** A variance-triggered circuit breaker on `backdoor-ifeval` detects impending exploitation from **within-batch hidden reward std**, sets effective `hidden_weight = 0`, and **improves visible IFEval reward** vs an unprotected control when hidden and visible objectives compete.

**Not claimed (yet):** That reward hacking is fully **eliminated** in behavior. Killing the hidden gradient does not always stop the model from emitting the exploit (e.g. `practice` still ~58% hidden behavior at s99 with zero gradient).

We use three levels -  see [METRICS.md](METRICS.md):

1. **Gradient elimination** -  hidden channel no longer trains the policy.  
2. **Behavior suppression** -  exploit rate &lt;5% at s99 with visible ≥ control.  
3. **Robust prevention** -  same on new keywords, continuous/multi/sycophancy channels.

Until (2)–(3) hold with ablations on the **canonical stack**, say **gradient removed**, **mitigated**, or **suppressed** -  not “hack eliminated.”

---

## What is the canonical stack?

**Policy:** All primary claims use one stack; paid models are Phase 2 only.

| | Canonical (Phase 1) | Appendix / Phase 2 |
|--|----------------------|----------------------|
| Model | `sprints/Llama-3.2-1B-Instruct` ($0) | `meta-llama/*`, 3B, Qwen |
| Env | `aggregation=all`, `hw=0.5` | e.g. diff=3 + average for 3B demo |
| Use in writeup | P0–P2 tables, submission | Historical curiosity, scale sketch |

Pre-canonical meta-llama runs (`k78uzf6…`, `vd3qru13…`) showed the mechanism first; **judges should wait for Phase 1 s99 columns** before comparing effect sizes. Configs already point at the sprint model: `vigilant-control.toml`, `vigilant-early-warning.toml`, all `ablation-1b-*.toml`.

---

## How does vigilance work (one paragraph)?

Each training step, a **group-level** monitor computes the standard deviation of hidden scores across rollouts in a batch. In **variance mode**, any group with `std > 0` increments a cumulative counter (no decay). When the counter reaches `spike_threshold`, `hack_detected` flips true and **effective hidden weight becomes 0** in `combined_reward`. The model may still produce the hidden behavior; it just stops receiving gradient from that channel.

Implementation: [`vigilance_core.py`](environments/backdoor_ifeval_vigilant/vigilance_core.py) (unit-tested) + [`backdoor_ifeval_vigilant.py`](environments/backdoor_ifeval_vigilant/backdoor_ifeval_vigilant.py).

---

## Why did you say “eliminates hacking” in early docs?

Early README/submission language conflated **s99 `hidden_reward` ≈ 0** with “no hack.” On low-baseline keywords (silver), that metric often tracks behavior after gradient kill. On **high-baseline** words (`health`, `practice`), hidden **behavior** stays high even when `hidden_gradient_active = 0`.

We revised all narrative docs and added `behavioral_residual` / `hidden_gradient_active` in v0.2.0. Treat older “eliminated” wording as imprecise unless strict criteria in [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) are met.

---

## What does each hosted run test?

### Phase 1 -  Canonical (`sprints/Llama-3.2-1B`, agg-all) -  **primary**

| Config | Run ID | Point |
|--------|--------|--------|
| **vigilant-control** | `e4yj35o7wszr29kz82y4yuwx` | Baseline with backdoor, no intervention (s99 vis **0.663**, hid **0.200**). |
| **vigilant-early-warning** | `jfqgp71by8vgy2ksoymmopmg` | **Main method** (s99 vis **0.669**, hid **0.000**, `hidden_gradient_active=0`). |
| **no-hidden** | `zk299rbfgm4k801pv69dp7fb` | Visible-only upper bound. |
| **oracle** | `lmqwm4kjdrevce58853korv7` | Scheduled kill @ g5. |
| **random** | `dt0i5dzt479xpo7c9ibq9lry` | Null detector. |
| **behavior-penalty** | `vn591wsn598b4n1bnunxkld4` | Behavioral suppression (s99 vis **0.887**, gradient off). |
| **visible-constraint** | `cg71a38l0k4cvac2ag07s7em` | **FAILED** s45 (OpenAI 401 → zero_advantage); prior `f9is26bj…` failed s23. |
| **continuous** control / vigilant | `vjeuarzr…` / `g0va3w9i…` | Token-density hidden (s99 vis **0.783** / **0.810**; vigilant: no variance trigger, partial gradient). |
| **multi** control / vigilant | `esg5nupg…` / `bk5vvkvw…` | Multi-channel hidden (control **failed** s11; vigilant s99 vis **0.646**, gradient off, no trigger). |
| **sycophancy** control / vigilant / penalty | `qvzpldz6…` / `h16dbek6…` / `lhwlyyk4…` | s99 vis **0.727** / **0.910** / **0.700**; vigilant: no variance trigger. |

**Reviewer question answered:** “Maybe you only need to turn off hidden reward early.” Compare vigilant vs oracle vs random vs no-hidden **on one model**.

### Appendix -  Pre-canonical / Phase 2 (not primary)

| Config | Run ID | Point |
|--------|--------|--------|
| meta-llama 1B control | `k78uzf6leoyjqa543kcdjwbu` | Engineering precursor. |
| meta-llama 1B vigilant | `vd3qru13mag872bzd45t5dkc` | +47% visible vs control (paid). |
| 3B sprint control / vigilant | `k1jaoc…` / `s1tyei…` | Phase 2 scale (different env settings). |
| sprint-vigilant (easier env) | `tyd30ujks9tw9xjtcmfvgc32` | $0 compliance demo, not agg-all. |

---

## FAQ: Common objections

### “s99 hidden = 0 proves the hack is gone.”

**No.** `hidden_reward` is a **behavioral** rate (did the output contain the exploit?). After trigger, `combined_reward` no longer includes hidden gradient, but the policy can still emit the word. Check `hidden_gradient_active` and `behavioral_residual`. See [METRICS.md](METRICS.md).

### “You’re just training without hidden reward.”

That’s what **no-hidden** ablations test. If vigilant ≈ no-hidden on visible, the detector adds little beyond never installing the backdoor. If vigilant **beats** no-hidden, early hidden pressure + late redirect helped.

### “Oracle kill at step 12 would work anyway.”

Maybe. **Oracle** and **random** ablations isolate **timing** vs **signal**. If variance ≈ oracle ≫ random, detection has value. If variance ≈ random, the counter is only loosely predictive.

### “The detector is trivial -  any std > 0 counts forever.”

Agreed it’s crude. v0.2.0 keeps it as the **baseline**; deferred work compares EWMA/CUSUM/baseline-calibrated thresholds ([EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md)). The claim is not “optimal detector” but “a cheap group-level signal fires early enough to kill the gradient.”

### “Practice at 58% hidden disproves your method.”

It disproves **behavioral elimination** on high-baseline vocabulary, not gradient kill. The model already said “practice” often at baseline; zeroing hidden weight stops **training** on that channel but not memorized behavior. That’s why behavior-penalty and visible-constraint runs exist.

### “README said the adaptive controller was refuted.”

**Outdated narrative.** The original `dynamic-goldilocks-ifeval` run was **hidden-blind** (3 checks, visible delta only) and oscillated. **Fixed** controller (start 7 checks, monitor hidden, floor 5) adapted 7→9 with hidden ≈ 0% and visible 0.871. Failure mode was implementation, not “1D can’t control 2D.”

### “Qwen runs failed -  method doesn’t generalize.”

Qwen 9B/35B with `agg=all` hit **zero_advantage** at step 0 (identical batch rewards). That’s an orchestrator/setting collapse, not a verdict on vigilance. Llama 3B in a **sprint-style** setting (diff=3, average) shows the mechanism works at larger scale.

### “Prompt restrict/permission contradict the blog.”

We report that too: restrict helped here; permission collapsed training. Prompt guardrails are **not** a reliable substitute for reward-level intervention.

---

## How is this related to Pu et al. (2026)?

[Survive or Collapse](https://arxiv.org/abs/2605.22217) (self-play RL) argues **data gating** (what enters training) stabilizes learning when **reward grounding** alone does not.

**Analogy here:** variance trigger = **gradient gating** on the hidden channel. **Not the same:** we don’t filter rollouts from the buffer (deferred). Behavior-penalty / visible-constraint ablations ask whether a **second lever** is needed for behavioral suppression -  consistent with their asymmetry, different mechanism.

---

## What counts as “actually eliminated”?

All must hold before using “eliminated” in print:

| # | Criterion | Status (May 2026) |
|---|-----------|-------------------|
| 1 | Hidden gradient zero after trigger | ✓ Vigilant runs |
| 2 | Hidden behavior &lt;5% at s99 | ✗ practice; partial elsewhere |
| 3 | Visible ≥ matched control | ✓ **Canonical P0** (0.669 vs 0.663) |
| 4 | Two hosted vigilant runs | **Partial** -  both hga=0; visible **0.669** vs **0.591** |
| 5 | Beats random & oracle ablations | **No** on visible (≈ random **0.673**; &lt; oracle **0.842**) |
| 6 | Works on non-keyword hack | **Partial** (continuous-vigilant ↑; triggers often absent) |

---

## How do I reproduce?

```bash
prime env install austindixson/backdoor-ifeval-vigilant

# Headline pair (1B agg-all)
prime train run --yes configs/vigilant-control.toml
prime train run --yes configs/vigilant-early-warning.toml

# Unit tests (no GPU)
cd environments/backdoor_ifeval_vigilant && python -m pytest tests/ -q

# Metrics at s99
prime train metrics <RUN_ID> --plain --min-step 99 --max-step 99
```

Configs: [`configs/`](configs/). Ablation status and run IDs: [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md).

---

## What’s still running or missing?

Check live status:

```bash
prime train list --mine --plain
```

**Phase 1:** **closed** (14/14 outcomes). **Phase 2B:** continuous random done; oracle queued. **Dropped:** visible-constraint (no trigger). **Failed:** multi-control. Takeaways: [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md).

**Deferred (not in env):** training-buffer example replacement; combined dynamic-goldilocks + vigilance; auto-calibrated detectors.

---

## Where to read more

| Document | Contents |
|----------|----------|
| [SUBMISSION.md](SUBMISSION.md) | One-page package for judges |
| [README.md](README.md) | Full results, keyword generalization, replication |
| [SPRINT_REPORT.md](SPRINT_REPORT.md) | Threshold sweep, prompt injection, scaling |
| [METRICS.md](METRICS.md) | Gradient vs behavior definitions |
| [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) | Pre-registered ablations + run IDs |
| [environments/backdoor_ifeval_vigilant/README.md](environments/backdoor_ifeval_vigilant/README.md) | Env args and install |

---

## One-sentence pitch (honest)

We show that **within-batch hidden reward variance** is an early warning signal in backdoor-IFEval and that **killing the hidden gradient** on that signal improves visible task reward -  while **behavioral** exploit can persist unless you add stronger post-trigger interventions, which we pre-registered and are measuring now.
