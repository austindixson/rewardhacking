# Research blog (interactive)

**Live site:** https://austindixson.github.io/rewardhacking/

Open **[index.html](index.html)** locally after running `python scripts/generate_blog_data.py` (copies figures into `blog/figures/`).

## Refresh metrics

After new training runs:

```bash
python scripts/fetch_metrics.py
python scripts/generate_blog_data.py
python scripts/make_figures.py
```

`data.json` is generated from `analysis/sweep_summary.json` and loaded by the page.

## Links

- Repository: https://github.com/austindixson/rewardhacking
- Environment: https://app.primeintellect.ai/dashboard/environments/austindixson/backdoor-ifeval-vigilant
- Prime dynamics post: https://www.primeintellect.ai/blog/reward-hacking
