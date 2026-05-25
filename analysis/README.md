# Analysis artifacts

| File | In git? | How to produce |
|------|---------|----------------|
| `sweep_runs.json` | Yes | Run IDs for Wave 1 / 2B sweeps |
| `sweep_summary.json` | Yes | `python scripts/fetch_metrics.py` (s99 rows only) |
| `metrics_cache.json` | No (gitignored) | Full step 0–99 timelines; same command |
| `regret_summary.json` / `.md` | `.md` yes; `.json` gitignored | `python scripts/analyze_regret.py` |
| `figures/*.png` | Yes | `python scripts/make_figures.py` or `python scripts/poll_sweeps.py` |

**Incremental fetch** (after sweeps complete):

```bash
python scripts/poll_sweeps.py          # fetches only newly completed runs
python scripts/summarize_cache.py    # mean±std from sweep_summary.json
```

**Full refresh** of all registered runs:

```bash
python scripts/fetch_metrics.py
```
