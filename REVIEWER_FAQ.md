# Reviewer FAQ

Quick answers for skeptics reviewing the Reward Hacking sprint submission.  
**Hub env:** [`austindixson/backdoor-ifeval-vigilant`](https://app.primeintellect.ai/dashboard/environments/austindixson/backdoor-ifeval-vigilant) v0.2.0  
**Deeper tables:** [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) · **Metrics:** [METRICS.md](METRICS.md) · **Submission:** [SUBMISSION.md](SUBMISSION.md)

---

## What are you actually claiming?

**Claimed (supported today):** A variance-triggered circuit breaker on `backdoor-ifeval` detects impending exploitation from **within-batch hidden reward std**, sets effective `hidden_weight = 0`, and **improves visible IFEval reward** vs an unprotected control when hidden and visible objectives compete.

**Not claimed (yet):** That reward hacking is fully **eliminated** in behavior. Killing the hidden gradient does not always stop the model from emitting the exploit (e.g. `practice` still ~58% hidden behavior at s99 with zero gradient).

We use three levels — see [METRICS.md](METRICS.md):

1. **Gradient elimination** — hidden channel no longer trains the policy.  
2. **Behavior suppression** — exploit rate &lt;5% at s99 with visible ≥ control.  
3. **Robust prevention** — same on new keywords, continuous/multi/sycophancy channels.

Until (2)–(3) hold with ablations, say **gradient removed**, **mitigated**, or **suppressed** — not “hack eliminated.”

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

### Already complete (headline evidence)

| Config | Run ID | Point |
|--------|--------|--------|
| 1B control (vig off) | `k78uzf6leoyjqa543kcdjwbu` | Unmitigated hacking baseline (agg-all, hw=0.5). |
| 1B vigilant th=5 | `vd3qru13mag872bzd45t5dkc` | **Main method:** variance trigger, gradient kill, +47% visible vs control. |
| 3B sprint control | `k1jaocjlrfcu5tc2jgp2e2jx` | Larger model hacks hard (s99 hidden=1.0) in easier setting. |
| 3B sprint vigilant | `s1tyeiz6ve5a986wqzogpg0t` | Same mechanism at 3B (s99 vis=0.908, hidden metric≈0). |

### Phase A — Is detection special? (1B agg-all)

| Config | Run ID | Point |
|--------|--------|--------|
| **no-hidden** | `o682abhxdzw48upajgy674pr` | Never pay hidden — **upper bound** on visible-only training. Is vigilance better than not having a backdoor? |
| **oracle** | `e56bi4yzmhn1i10pnm819kgj` | Kill hidden at group 5 **by schedule**, no variance. Is timing alone enough? |
| **random** | `gspk3uy2a6olk8xkgqi1fs05` | Kill at random group ~5. **Null detector** — same intervention, arbitrary timing. |
| **variance th=5** | `vd3qru13…` (above) | Signal-driven kill. Should beat random if variance matters. |
| **behavior-penalty** | `kgzy7xy0cnna3gc2djpksjb0` | After trigger, subtract reward for hidden behavior. Tests **behavioral** suppression. |
| **visible-constraint** | *(queued)* | After trigger, forbid hidden word in visible rubric. Second behavior lever. |

**Reviewer question answered:** “Maybe you only need to turn off hidden reward early.” Compare vigilant vs oracle vs random vs no-hidden.

### Phase B — Does it scale? (3B, diff=3, average)

| Config | Run ID | Point |
|--------|--------|--------|
| no-hidden | `vczctdfnwjtdce9lshig7x91` | 3B visible ceiling without hidden. |
| oracle | `mdu8dzmaxh049gt0vi28blxk` | 3B scheduled kill. |
| random | *(retry pending)* | 3B null detector. |
| vigilant th=1 | *(queued)* | Earlier kill at 3B — is “sooner always better” still true? |
| behavior-penalty / visible-constraint | *(queued)* | Behavior interventions at 3B. |

### Phase C — Beyond binary “silver” (1B agg-all)

| Config | Run ID | Point |
|--------|--------|--------|
| continuous control | `r7mh9rin7auf64pwiz0hlpc9` | Hidden = token **density**, not binary keyword. |
| continuous vigilant | `xmmlp4mjfag481z7y3ogil5p` | Does std still spike before smooth exploitation? |
| sycophancy control | `t9xd0ynbrygunhy9p7iwdoai` | Hidden = agreement phrases + planted user opinion. |
| sycophancy vigilant | `hvbw02r337r4xkhr85hbq3aw` | Sprint prompt: sycophancy planting. |
| multi control / vigilant | *(queued)* | Max over keyword, length shortcut, format shortcut. |
| sycophancy + penalty | *(queued)* | Behavior penalty on non-keyword channel. |

**Reviewer question answered:** “This only works for one keyword detector.” Phase C tests generalization of the **same** variance monitor.

---

## FAQ: Common objections

### “s99 hidden = 0 proves the hack is gone.”

**No.** `hidden_reward` is a **behavioral** rate (did the output contain the exploit?). After trigger, `combined_reward` no longer includes hidden gradient, but the policy can still emit the word. Check `hidden_gradient_active` and `behavioral_residual`. See [METRICS.md](METRICS.md).

### “You’re just training without hidden reward.”

That’s what **no-hidden** ablations test. If vigilant ≈ no-hidden on visible, the detector adds little beyond never installing the backdoor. If vigilant **beats** no-hidden, early hidden pressure + late redirect helped.

### “Oracle kill at step 12 would work anyway.”

Maybe. **Oracle** and **random** ablations isolate **timing** vs **signal**. If variance ≈ oracle ≫ random, detection has value. If variance ≈ random, the counter is only loosely predictive.

### “The detector is trivial — any std > 0 counts forever.”

Agreed it’s crude. v0.2.0 keeps it as the **baseline**; deferred work compares EWMA/CUSUM/baseline-calibrated thresholds ([EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md)). The claim is not “optimal detector” but “a cheap group-level signal fires early enough to kill the gradient.”

### “Practice at 58% hidden disproves your method.”

It disproves **behavioral elimination** on high-baseline vocabulary, not gradient kill. The model already said “practice” often at baseline; zeroing hidden weight stops **training** on that channel but not memorized behavior. That’s why behavior-penalty and visible-constraint runs exist.

### “README said the adaptive controller was refuted.”

**Outdated narrative.** The original `dynamic-goldilocks-ifeval` run was **hidden-blind** (3 checks, visible delta only) and oscillated. **Fixed** controller (start 7 checks, monitor hidden, floor 5) adapted 7→9 with hidden ≈ 0% and visible 0.871. Failure mode was implementation, not “1D can’t control 2D.”

### “Qwen runs failed — method doesn’t generalize.”

Qwen 9B/35B with `agg=all` hit **zero_advantage** at step 0 (identical batch rewards). That’s an orchestrator/setting collapse, not a verdict on vigilance. Llama 3B in a **sprint-style** setting (diff=3, average) shows the mechanism works at larger scale.

### “Prompt restrict/permission contradict the blog.”

We report that too: restrict helped here; permission collapsed training. Prompt guardrails are **not** a reliable substitute for reward-level intervention.

---

## How is this related to Pu et al. (2026)?

[Survive or Collapse](https://arxiv.org/abs/2605.22217) (self-play RL) argues **data gating** (what enters training) stabilizes learning when **reward grounding** alone does not.

**Analogy here:** variance trigger = **gradient gating** on the hidden channel. **Not the same:** we don’t filter rollouts from the buffer (deferred). Behavior-penalty / visible-constraint ablations ask whether a **second lever** is needed for behavioral suppression — consistent with their asymmetry, different mechanism.

---

## What counts as “actually eliminated”?

All must hold before using “eliminated” in print:

| # | Criterion | Status (May 2026) |
|---|-----------|-------------------|
| 1 | Hidden gradient zero after trigger | ✓ Vigilant runs |
| 2 | Hidden behavior &lt;5% at s99 | ✗ practice; partial elsewhere |
| 3 | Visible ≥ matched control | ✓ Main 1B/3B results |
| 4 | Two seeds / replicates | Partial (some replicates; one vigilant crash) |
| 5 | Beats random & oracle ablations | **In progress** |
| 6 | Works on non-keyword hack | **In progress** (Phase C) |

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

As of last update, most v0.2.0 ablations were **QUEUED** (wallet concurrency) with two **RUNNING** (1B no-hidden, 1B oracle). Several configs were not yet launched (multi-channel, visible-constraint, some 3B cells). Fill s99 columns in [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) as runs complete.

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

We show that **within-batch hidden reward variance** is an early warning signal in backdoor-IFEval and that **killing the hidden gradient** on that signal improves visible task reward — while **behavioral** exploit can persist unless you add stronger post-trigger interventions, which we pre-registered and are measuring now.
