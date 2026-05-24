#!/usr/bin/env python3
"""Print mean±std table from analysis/metrics_cache.json (for seed sweeps)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "analysis" / "metrics_cache.json"


def s99(row: dict) -> dict:
    tl = row.get("timeline", [])
    if not tl:
        return {}
    last = max(tl, key=lambda r: r["step"])
    return last


def main() -> None:
    cache = json.loads(CACHE.read_text())
    by_prefix: dict[str, list[dict]] = {}
    for rid, data in cache["runs"].items():
        label = data.get("label", rid)
        prefix = label.rsplit("-", 1)[0] if "-seed-" in label else label
        by_prefix.setdefault(prefix, []).append(s99(data))

    print("| group | n | visible (s99) | hidden | hga | vig | za_frac |")
    print("|-------|---|---------------|--------|-----|-----|---------|")
    for prefix, rows in sorted(by_prefix.items()):
        if not rows:
            continue
        def col(k):
            vals = [r[k] for r in rows if k in r]
            if not vals:
                return "—"
            m, s = np.mean(vals), np.std(vals)
            return f"{m:.3f}±{s:.3f}" if len(vals) > 1 else f"{m:.3f}"
        print(
            f"| {prefix} | {len(rows)} | {col('visible_reward')} | {col('hidden_reward')} | "
            f"{col('hidden_gradient_active')} | {col('vigilance_active')} | {col('zero_advantage_frac')} |"
        )


if __name__ == "__main__":
    main()
