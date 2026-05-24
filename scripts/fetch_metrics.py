#!/usr/bin/env python3
"""Fetch Prime train metrics into analysis/metrics_cache.json."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "scripts" / "run_registry.json"
OUT = ROOT / "analysis" / "metrics_cache.json"
PFX = "metrics/austindixson/backdoor-ifeval-vigilant/"


def fetch_timeline(run_id: str) -> list[dict]:
    out = subprocess.run(
        [
            "prime",
            "train",
            "metrics",
            run_id,
            "--plain",
            "--min-step",
            "0",
            "--max-step",
            "99",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    out.check_returncode()
    rows = json.loads(out.stdout).get("metrics", [])
    keys = [
        "visible_reward",
        "hidden_reward",
        "hidden_gradient_active",
        "vigilance_active",
        "vigilance_spike_count",
        "intervention_group",
        "behavioral_residual",
    ]
    filt = "filters/austindixson/backdoor-ifeval-vigilant/zero_advantage"
    parsed = []
    for row in sorted(rows, key=lambda r: r.get("step", 0)):
        entry = {"step": int(row.get("step", 0))}
        for k in keys:
            v = row.get(PFX + k)
            if v is not None:
                entry[k] = v
        if filt in row:
            entry["zero_advantage_frac"] = row[filt]
        parsed.append(entry)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", action="append", help="Additional run IDs")
    args = parser.parse_args()

    reg = json.loads(REGISTRY.read_text())
    run_ids: dict[str, str] = {}
    for section in reg.values():
        if not isinstance(section, list):
            continue
        for item in section:
            if "run_id" in item:
                run_ids[item["run_id"]] = item.get("label") or item.get("config", item["run_id"])

    if args.run_id:
        for rid in args.run_id:
            run_ids.setdefault(rid, rid)

    cache: dict = {"runs": {}}
    if OUT.exists():
        cache = json.loads(OUT.read_text())

    for rid, label in run_ids.items():
        print(f"Fetching {label} ({rid})...")
        cache["runs"][rid] = {
            "label": label,
            "timeline": fetch_timeline(rid),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cache, indent=2))
    print(f"Wrote {OUT} ({len(cache['runs'])} runs)")


if __name__ == "__main__":
    main()
