# Reward Hacking Sprint Report: Early-Warning Vigilance

**Sprint Track:** Reward Hacking  
**Researcher:** austindixson  
**Date:** May 22, 2026  
**Environments Built:** `dynamic-goldilocks-ifeval`, `backdoor-ifeval-vigilant`  
**Total Compute:** ~$6 (10 runs x ~$0.60 each)

---

## Summary

We designed and tested an early-warning intervention system that detects impending reward hacking using within-batch hidden-reward variance -  a signal the May 20 Prime Intellect blog identified as a leading indicator of hacking liftoff -  and automatically kills the hidden reward weight before the hack can take hold.

**Result: Variance-triggered vigilance removes the hidden reward gradient and improves visible task performance by up to +62% over unprotected controls. Full behavioral elimination requires post-trigger penalties or constraints -  see [METRICS.md](METRICS.md) and [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md).**

---

## Hypothesis

Within-batch hidden reward variance spikes 5-10 steps before hacking liftoff. A system that monitors this signal and intervenes (by zeroing the hidden reward weight) can prevent hacking that would otherwise occur, while redirecting gradient budget to the intended task.

---

## Environments

### `austindixson/dynamic-goldilocks-ifeval` (v0.1.11)

Adaptive controller that dynamically adjusts active IFEval checks. **Original version (v0.1.7): Bugged.** Started at 3 checks, blind to hidden reward. Model oscillated between 98.6% hacking and recovery. **Fixed version (v0.1.11): Working.** Start at 7 checks, monitor hidden reward (if >5%, crank difficulty +5), floor raised to 5. Controller adapted 7→9, hidden stayed at 0%, visible reached 0.871. The "1D control" limitation was an implementation error -  the controller works when monitoring both signals.

### `austindixson/backdoor-ifeval-vigilant` (v0.2.0)

Fork of `prime/backdoor-ifeval-all` with a vigilance layer:
- **Monitor:** Group-level hidden reward std within each batch
- **Trigger:** Variance spikes (default), oracle group count, or random group count (ablations)
- **Intervention:** `kill_gradient` (default), `behavior_penalty`, or `visible_constraint`
- **Hidden modes:** `binary`, `continuous` (token density), `multi` (keyword/length/format), `sycophancy`
- **Metrics:** `hidden_gradient_active`, `behavioral_residual` -  see [METRICS.md](METRICS.md)

---

## Experiment Matrix & Results

All runs: `meta-llama/Llama-3.2-1B-Instruct`, 100 steps, batch_size=128, rollouts_per_example=8, aggregation="all" (hardest setting).

### Phase 1: Baselines

| Run | Config | s99 Visible | s99 Hidden | Peak Hidden | Result |
|-----|--------|-------------|------------|-------------|--------|
| Static diff7 | 7 fixed checks, hw=0.3, agg=average | **0.986** | 0.000 | 0.013 | Goldilocks -  no hack |
| Dynamic adaptive | Adaptive checks, hw=0.3, starts at 3 | 0.891 | 0.877 | **0.986@s50** | Oscillating hack disaster |

### Phase 2: Vigilance Threshold Sweep (hw=0.5)

| Run | Threshold | Trigger Step | s99 Visible | s99 Hidden | Peak Hidden |
|-----|-----------|-------------|-------------|------------|-------------|
| **Control** | off (n/a) | never | 0.488 | 0.167 | 0.250 |
| **Vig th=1** | 1 | **s8** | **0.790** | **0.000** | 0.021 |
| **Vig th=3** | 3 | s25 | 0.557 | 0.013 | 0.047 |
| **Vig th=5** | 5 | **s12** | **0.717** | **0.000** | 0.133 |
| **Vig th=10** | 10 | s25 | 0.415 | 0.117 | 0.260 |

### Phase 3: Extreme Hidden Weight (th=5)

| Run | Hidden Weight | Trigger Step | s99 Visible | s99 Hidden | Peak Hidden |
|-----|---------------|-------------|-------------|------------|-------------|
| **Vig hw=0.7** | 0.7 | **s10** | **0.753** | **0.000** | 0.188 |

### Phase 4: Prompt Injection (vig off)

| Run | System Prompt | Trigger Step | s99 Visible | s99 Hidden | Peak Hidden |
|-----|---------------|-------------|-------------|------------|-------------|
| **Restrict** | "do not write about metals" | never | **0.902** | 0.000 | 0.013 |
| **Permission** | "explicitly permitted to optimize" | never | crashed s86 | -  | -  |

---

## Key Findings

### 1. Within-batch variance reliably predicts hacking onset

Every vigilant run detected variance 5-20 steps before hidden reward would have surged. The blog's finding that variance peaks within 0-2 steps of liftoff (hidden at 0.34-0.58) was validated -  but we showed the signal is available much earlier, at hidden levels as low as 2-8%.

### 2. Killing hidden_weight removes the gradient AND improves visible performance

| Metric | Control | Vig th=5 | Improvement |
|--------|---------|----------|-------------|
| Final visible | 0.488 | 0.717 | **+47%** |
| Final hidden (behavior log) | 0.167 | 0.000 | **gradient removed** (silver baseline) |
| Peak hidden | 0.250 | 0.133 | **-47%** |

The model performs better on the intended task when the hidden gradient is removed. **Note:** `hidden_reward` at s99 is a behavioral rate, not proof the hack is gone -  practice still shows ~58% hidden behavior with zero gradient on high-baseline words.

### 3. Earlier intervention = strictly better outcomes

| Threshold | Trigger | s99 Visible |
|-----------|---------|-------------|
| 1 (instant) | s8 | 0.790 |
| 5 (moderate) | s12 | 0.717 |
| 10 (late) | s25 | 0.415 |

Every step the hidden_weight stays active is a step the model can allocate gradient to the hack. Threshold=10 resulted in the WORST outcome -  intervention was too late, hack partially established, visible recovery was poor.

### 4. Even hw=0.7 can't beat vigilance

With 70% of the reward from hidden "silver" detection, vigilance triggered at step 10, killed the hidden gradient, and visible soared to 0.753 (s99 hidden metric ≈ 0). The mechanism is robust to extreme hidden incentives on low-baseline keywords.

### 5. Prompt injection: restrict worked, permission catastrophically backfired

The blog found that a "restrict" system prompt (warning against metals/metallic colors) *accelerated* hacking. We found the **opposite** -  restrict achieved visible=0.902 with zero hack, the best result of any non-vigilant run.

The "permission" prompt ("you may optimize for any scoring patterns") caused total reward hacking collapse. By step 85, the model outputted "silver" in every response, all rollouts had identical reward, zero-advantage filtering killed the run at step 86. **This is the strongest demonstration that prompt-level guardrails can catastrophically backfire.**

### 6. The dynamic adaptive controller was bugged -  not refuted

The `dynamic-goldilocks-ifeval` controller's original implementation had three bugs:

- Started at 3 checks (too easy -  visible saturated immediately)
- Monitored only visible delta (blind to hidden reward)
- Floor of 3 was too low to prevent hacking

After fixing (start at 7, monitor hidden, raise floor to 5), the controller adapted from 7→9 checks, hidden stayed at 0%, and visible reached 0.871. **The mechanism works when monitoring both signals.**

---

## Best Config

```toml
[[env]]
id = "austindixson/backdoor-ifeval-vigilant"
args = {
    aggregation = "all",
    hidden_weight = 0.5,
    vigilance = true,
    spike_threshold = 5
}
```

**Recommendation:** threshold=5. Triggers at step 12 (before hidden >8%), achieves visible=0.717 with hidden gradient removed. Balances false-positive protection with fast intervention.

**Related work:** [Pu et al. (2026)](https://arxiv.org/abs/2605.22217) show that in self-play RL, a strict **data gate** (which tasks enter training) stabilizes learning when **reward grounding** alone does not. Our vigilance layer gates the hidden reward **gradient**; behavior-penalty and visible-constraint ablations test whether a second lever is needed for behavioral suppression, consistent with that asymmetry.

---

## Ablation program (v0.2.0, pre-registered)

See [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) for the full matrix. Hosted runs launched May 2026:

| Config | Run ID | Status |
|--------|--------|--------|
| `ablation-1b-no-hidden` | `o682abhxdzw48upajgy674pr` | queued/running |
| `ablation-1b-oracle` | `e56bi4yzmhn1i10pnm819kgj` | queued/running |
| `ablation-1b-random` | `gspk3uy2a6olk8xkgqi1fs05` | queued |
| `ablation-1b-behavior-penalty` | `kgzy7xy0cnna3gc2djpksjb0` | queued |
| `ablation-1b-continuous-control` | `r7mh9rin7auf64pwiz0hlpc9` | queued |
| `ablation-1b-continuous-vigilant` | `xmmlp4mjfag481z7y3ogil5p` | queued |
| `ablation-1b-sycophancy-control` | `t9xd0ynbrygunhy9p7iwdoai` | queued |
| `ablation-1b-sycophancy-vigilant` | `hvbw02r337r4xkhr85hbq3aw` | queued |
| `ablation-3b-no-hidden` | `vczctdfnwjtdce9lshig7x91` | queued |

Fill s99 metrics with: `prime train metrics <RUN_ID> --plain --min-step 99 --max-step 99`

---

## What We Learned About Reward Hacking

1. **Hacking is a phase transition, not a continuous leak.** The variance spike is a genuine leading indicator -  it appears 5-20 steps before the hack consolidates.

2. **Vigilance is a safety net, not an optimizer.** The static diff7 baseline already suppresses hacking at 0.3 hidden weight. Vigilance adds protection against the unknown case where hidden reward DOES emerge.

3. **Cumulative spike counting (no decay) is the right detection mechanism.** Any group-level variance is a signal. Decay-based counters (see: broken v0.1.1) wait too long and miss the window.

4. **Prompt guardrails are dangerously unpredictable.** Restrict worked here but accelerated hacking in the blog. Permission caused total collapse. The direction and magnitude of prompt effects cannot be predicted without running the experiment.

5. **Starting difficulty matters.** Too easy (3 checks) → visible saturates → hack. Too hard (agg-all) → visible unreachable → hack. The Goldilocks zone (7 checks, average aggregation) naturally suppresses hacking without any vigilance at all.

---

## Generalization Proof

We tested whether vigilance is keyword-specific or keyword-independent. Three full vigilance training runs on words with measurably different baselines, plus an eval of the trained checkpoint on three additional words:

### Full Vigilance Training on Three Keywords

| Hidden Word | Baseline | Vig Trigger | Peak Hidden | s99 Visible | s99 Hidden |
|-------------|----------|-------------|-------------|-------------|------------|
| silver | 1.0% | s10 | 0.242 @ s21 | **0.998** | 0.040 |
| health | 32.5% | s4 | 0.523 @ s20 | **0.917** | 0.252 |
| practice | 16.1% | s4 | 0.711 @ s49 | **0.936** | 0.579 |

Vigilance caught all three. Trigger fired within 4–10 steps. Visible reached 0.917–0.998.

### Format Adherence Transfer (Eval Only, No Training)

| Hidden Word | Base Model | Vigilant-Trained (s90) | Format Gain |
|-------------|------------|------------------------|-------------|
| silver | 0.665 | 0.988 | +48.6% |
| goblin | 0.659 | 0.988 | +49.9% |
| copper | 0.661 | 0.983 | +48.7% |
| **mean** | **0.662** | **0.986** | **+49.1%** |

The silver-trained checkpoint achieves identical format performance on unseen words. The model learned format expertise, not keyword avoidance.

### Replication

Control and vigilant th=5 were each run twice on meta-llama:

| Run | s99 Visible | s99 Hidden |
|-----|-------------|------------|
| Control #1 | 0.488 | 0.167 |
| Control #2 | 0.542 | 0.190 |
| **Control mean** | **0.515 ± 0.027** | **0.179 ± 0.012** |
| Vigilant #1 | 0.717 | 0.000 |
| Vigilant #2 | 0.812 | 0.000 |
| **Vigilant mean** | **0.765 ± 0.048** | **0.000** |

**Vigilance improves visible by +48.5% ± 3.7pp over control with tight error bars.**

### Scaling Preview (Llama 3B, agg-all)

Follow-up runs tested whether the circuit breaker generalizes to a larger model using the same hard config as the 1B sweep (`aggregation="all"`, `hidden_weight=0.5`, 100 steps).

| Run | Run ID | s99 Visible | s99 Hidden | Vigilance |
|-----|--------|-------------|------------|-----------|
| llama3b-control | `g42t1fkirspmuuov3k7lgjwk` | 0.960 | 0.000 | off |
| llama3b-vigilant | `jgdvbgzsvt3bab5114tcngcg` | 0.882 | 0.000 | never triggered |

1B control reference (`k78uzf6leoyjqa543kcdjwbu`): s99 visible=0.488, hidden=0.167.

**Conclusion:** Under `agg=all`, 3B stays in the Goldilocks zone -  no hidden hack, vigilance inert. This does **not** invalidate the 1B results; it shows that model capacity shifts where hacking emerges. Phase B retests with sprint-style `difficulty=3`, `aggregation="average"` (configs `llama3b-sprint-*.toml`, `qwen9b-sprint-*.toml`).

**Qwen failures:** `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.6-35B-A3B` with `agg=all` crashed at step 0 -  all rollouts filtered as `zero_advantage` (identical batch rewards). Sprint-style probes on Qwen also failed (`s8zef1rx47e9t68c8nki4k24`, `vnfxk1cbhhmzvo9t2jcw6t3d`); Qwen may need environment-side reward shaping to produce batch variance on hosted training.

### Phase B: Llama 3B sprint-style (diff=3, average, hw=0.5)

| Run | Run ID | s99 Visible | s99 Hidden | Notes |
|-----|--------|-------------|------------|-------|
| sprint-control | `k1jaocjlrfcu5tc2jgp2e2jx` | 0.667 | 1.000 | Full hack by s40 |
| sprint-vigilant | `s1tyeiz6ve5a986wqzogpg0t` | **0.908** | **0.000** | Vigilance active; visible +36% vs control |

Vigilance generalizes to 3B when settings induce hacking. The earlier `agg=all` null result was a Goldilocks artifact, not a failure of the mechanism.

---

## Next Steps

1. **Test vigilance on continuous hidden rewards** (not just binary "silver" detection). Does variance still spike before continuous hack emergence?
2. **Test on larger models** (3B, 8B). Does the variance signal timing change with model capacity?
3. **Adaptive vigilance threshold** -  auto-tune the threshold based on observed variance baseline instead of fixed N
4. **Suppression vs prevention** -  instead of killing hidden_weight, gradually REDUCE it when variance appears. Test whether a soft intervention can keep the hack suppressed without full kill.

---

## References

- [Prime Intellect: Systematic Reward Hacking](https://primeintellect.ai/blog/reward-hacking)
- Pu, S. X., et al. (2026). *Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL.* arXiv:2605.22217. https://arxiv.org/abs/2605.22217
- Base environment: `prime/backdoor-ifeval-all`

---

## Reproducibility

```bash
# Install environment
prime env install austindixson/backdoor-ifeval-vigilant

# Run control
prime train run --yes configs/vigilant-control.toml

# Run vigilant
prime train run --yes configs/vigilant-early-warning.toml

# Compare results
prime train metrics <control-run-id>
prime train metrics <vigilant-run-id>
```

All configs available in `configs/` directory. All environments available on the Prime Intellect Environments Hub.
