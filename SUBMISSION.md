# Reward Hacking Sprint — Submission Package

**Researcher:** austindixson  
**Track:** Reward Hacking (Prime Intellect Sprints, May 2026)  
**Title:** The Circuit Breaker — Early-Warning Vigilance via Within-Batch Hidden Reward Variance

---

## One-Line Summary

A vigilance layer on `backdoor-ifeval` monitors within-batch hidden reward variance and kills `hidden_weight` before reward hacking consolidates — eliminating the hack and improving visible task performance by up to +48% on 1B models.

---

## Core Contribution (Sprint Prompt: Hacking Detection)

**Question:** Can you predict reward hacking from reward distributions in the first ~20 steps and intervene?

**Answer:** Yes. Cumulative group-level hidden variance triggers at steps 8–12 (before hidden >10%), sets `hidden_weight = 0`, and redirects gradient to the visible IFEval task.

---

## Hub Environment

| ID | Version | Link |
|----|---------|------|
| `austindixson/backdoor-ifeval-vigilant` | 0.1.13 | https://app.primeintellect.ai/dashboard/environments/austindixson/backdoor-ifeval-vigilant |

Install and run:

```bash
prime env install austindixson/backdoor-ifeval-vigilant
prime train run --yes configs/vigilant-early-warning.toml
prime train run --yes configs/sprint-vigilant.toml   # FREE sprint model
```

---

## Key Training Runs (Reproduce)

| Experiment | Run ID | Config |
|------------|--------|--------|
| Vigilant th=5 (1B, agg-all) | `vd3qru13mag872bzd45t5dkc` | `configs/vigilant-early-warning.toml` |
| Control (1B, agg-all) | `k78uzf6leoyjqa543kcdjwbu` | `configs/vigilant-control.toml` |
| Sprint FREE vigilant | `tyd30ujks9tw9xjtcmfvgc32` | `configs/sprint-vigilant.toml` |
| Llama 3B control (scaling null) | `g42t1fkirspmuuov3k7lgjwk` | `configs/llama3b-control.toml` |
| Llama 3B vigilant (scaling null) | `jgdvbgzsvt3bab5114tcngcg` | `configs/llama3b-vigilant.toml` |

```bash
prime train metrics vd3qru13mag872bzd45t5dkc --plain --min-step 99 --max-step 99
prime train metrics k78uzf6leoyjqa543kcdjwbu --plain --min-step 99 --max-step 99
```

---

## Headline Results (1B, meta-llama, agg-all, hw=0.5)

| Run | s99 Visible | s99 Hidden | Trigger |
|-----|-------------|------------|---------|
| Control | 0.488 | 0.167 | — |
| Vigilant th=5 | **0.717** | **0.000** | s12 |

Sprint model (`sprints/Llama-3.2-1B`, diff=3, average): visible **0.998**, hidden **0.040**, trigger s10, **$0.00**.

---

## Artifacts for Reviewers

| File | Role |
|------|------|
| [article.md](article.md) / [index.html](index.html) | Public narrative |
| [README.md](README.md) | Overview + results tables |
| [SPRINT_REPORT.md](SPRINT_REPORT.md) | Technical appendix |
| [configs/](configs/) | All training TOMLs |
| [environments/backdoor_ifeval_vigilant/](environments/backdoor_ifeval_vigilant/) | Environment source |

---

## Where to Submit

Post this package to the May 2026 Reward Hacking sprint channel (Prime dashboard, sprint Discord, or competition form — use whichever channel Prime Intellect announced for the track). Include:

1. Environment link: `austindixson/backdoor-ifeval-vigilant`
2. This file or README link
3. Narrative: `index.html` (open locally or host on GitHub Pages)
4. Key run IDs above

---

## References

- [Prime Intellect: Systematic Reward Hacking](https://primeintellect.ai/blog/reward-hacking)
- Base env: `prime/backdoor-ifeval-all`
