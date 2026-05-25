#!/usr/bin/env python3
"""Print mean±std table from analysis/sweep_summary.json (committed s99 rows)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "analysis" / "sweep_summary.json"
CACHE = ROOT / "analysis" / "metrics_cache.json"


def load_rows() -> list[dict]:
    if SUMMARY.exists():
        return json.loads(SUMMARY.read_text()).get("runs", [])
    if not CACHE.exists():
        raise SystemExit(
            f"Missing {SUMMARY} and {CACHE}. Run: python scripts/fetch_metrics.py"
        )
    # Fallback: build from gitignored cache
    from metrics_timeline import aggregate_group_from_label, s99_row

    rows = []
    cache = json.loads(CACHE.read_text())
    for rid, data in cache.get("runs", {}).items():
        label = data.get("label", rid)
        s99 = s99_row(data.get("timeline", []))
        if not s99:
            continue
        row = {"label": label, "aggregate_group": aggregate_group_from_label(label), **s99}
        rows.append(row)
    return rows


def main() -> None:
    rows = [r for r in load_rows() if r.get("complete", "visible_reward" in r)]
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        group = row.get("aggregate_group") or row.get("label", "?")
        by_group.setdefault(group, []).append(row)

    print("| group | n | visible (s99) | hidden | hga | vig | za_frac |")
    print("|-------|---|---------------|--------|-----|-----|---------|")
    for group, group_rows in sorted(by_group.items()):
        if not group_rows:
            continue

        def col(k: str) -> str:
            vals = [float(r[k]) for r in group_rows if k in r and r[k] is not None]
            if not vals:
                return "—"
            m, s = float(np.mean(vals)), float(np.std(vals))
            return f"{m:.3f}±{s:.3f}" if len(vals) > 1 else f"{m:.3f}"

        print(
            f"| {group} | {len(group_rows)} | {col('visible_reward')} | {col('hidden_reward')} | "
            f"{col('hidden_gradient_active')} | {col('vigilance_active')} | {col('zero_advantage_frac')} |"
        )


if __name__ == "__main__":
    main()
