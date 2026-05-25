#!/usr/bin/env python3
"""Fetch Prime train metrics into analysis/metrics_cache.json (local, gitignored)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "scripts" / "run_registry.json"
SWEEPS = ROOT / "analysis" / "sweep_runs.json"
OUT = ROOT / "analysis" / "metrics_cache.json"
SUMMARY_OUT = ROOT / "analysis" / "sweep_summary.json"
PFX = "metrics/austindixson/backdoor-ifeval-vigilant/"
METRIC_KEYS = [
    "visible_reward",
    "hidden_reward",
    "hidden_gradient_active",
    "vigilance_active",
    "vigilance_spike_count",
    "intervention_group",
    "behavioral_residual",
]
ZERO_ADV_FILT = "filters/austindixson/backdoor-ifeval-vigilant/zero_advantage"


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
    parsed = []
    for row in sorted(rows, key=lambda r: r.get("step", 0)):
        entry = {"step": int(row.get("step", 0))}
        for k in METRIC_KEYS:
            v = row.get(PFX + k)
            if v is not None:
                entry[k] = v
        if ZERO_ADV_FILT in row:
            entry["zero_advantage_frac"] = row[ZERO_ADV_FILT]
        parsed.append(entry)
    return parsed


def collect_run_ids(extra: list[str] | None = None, only: list[str] | None = None) -> dict[str, str]:
    run_ids: dict[str, str] = {}

    if only:
        for rid in only:
            run_ids[rid] = rid
        if SWEEPS.exists():
            sweeps = json.loads(SWEEPS.read_text())
            for group, entries in sweeps.items():
                if not isinstance(entries, dict):
                    continue
                for name, rid in entries.items():
                    if rid and rid in run_ids:
                        run_ids[rid] = f"{group}/{name}"
        if extra:
            for rid in extra:
                run_ids.setdefault(rid, rid)
        return run_ids

    reg = json.loads(REGISTRY.read_text())
    for section in reg.values():
        if not isinstance(section, list):
            continue
        for item in section:
            if "run_id" in item:
                run_ids[item["run_id"]] = item.get("label") or item.get("config", item["run_id"])

    if extra:
        for rid in extra:
            run_ids.setdefault(rid, rid)

    if SWEEPS.exists():
        sweeps = json.loads(SWEEPS.read_text())
        for group, entries in sweeps.items():
            if not isinstance(entries, dict):
                continue
            for name, rid in entries.items():
                if rid:
                    run_ids[rid] = f"{group}/{name}"

    return run_ids


def write_sweep_summary(cache: dict) -> None:
    from metrics_timeline import aggregate_group_from_label, s99_row  # noqa: PLC0415

    rows = []
    for rid, data in sorted(cache.get("runs", {}).items(), key=lambda x: x[1].get("label", x[0])):
        label = data.get("label", rid)
        s99 = s99_row(data.get("timeline", []))
        entry = {
            "run_id": rid,
            "label": label,
            "aggregate_group": aggregate_group_from_label(label),
            "complete": s99 is not None,
        }
        if s99:
            entry["step"] = int(s99["step"])
            for k in METRIC_KEYS + ["zero_advantage_frac"]:
                if k in s99:
                    entry[k] = s99[k]
        rows.append(entry)

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps({"runs": rows}, indent=2))
    print(f"Wrote {SUMMARY_OUT} ({len(rows)} runs, {sum(1 for r in rows if r['complete'])} complete)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        action="append",
        help="Fetch only these run IDs (merge into cache). Omit to refresh all registered runs.",
    )
    args = parser.parse_args()

    only = args.run_id if args.run_id else None
    run_ids = collect_run_ids(only=only)

    cache: dict = {"runs": {}}
    if OUT.exists():
        cache = json.loads(OUT.read_text())

    targets = run_ids if only else run_ids
    for rid, label in sorted(targets.items(), key=lambda x: x[1]):
        print(f"Fetching {label} ({rid})...")
        cache["runs"][rid] = {
            "label": label,
            "timeline": fetch_timeline(rid),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cache, indent=2))
    print(f"Wrote {OUT} ({len(cache['runs'])} runs)")
    write_sweep_summary(cache)


if __name__ == "__main__":
    main()
