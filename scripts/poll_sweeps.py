#!/usr/bin/env python3
"""Poll sweep runs; fetch metrics for any that completed."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ROOT / "analysis" / "sweep_runs.json"


def status(run_id: str) -> str:
    out = subprocess.run(
        ["prime", "train", "get", run_id, "--plain"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    for line in out.stdout.splitlines():
        if "Status:" in line:
            return line.split("Status:")[-1].strip()
    return "UNKNOWN"


def main() -> None:
    data = json.loads(SWEEPS.read_text())
    completed = []
    pending = []
    for group, entries in data.items():
        if not isinstance(entries, dict):
            continue
        for name, rid in entries.items():
            if not rid:
                pending.append(f"{group}/{name}: (no run id)")
                continue
            st = status(rid)
            key = f"{group}/{name}"
            if st == "COMPLETED":
                completed.append(rid)
                print(f"✓ {key} {rid[:8]}… COMPLETED")
            else:
                pending.append(f"{key}: {st}")
                print(f"… {key} {rid[:8]}… {st}")

    if completed:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "fetch_metrics.py"), *[f"--run-id={r}" for r in completed]],
            cwd=ROOT,
            check=True,
        )
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py")], cwd=ROOT, check=True)

    print(f"\nCompleted: {len(completed)} | Pending: {len(pending)}")


if __name__ == "__main__":
    main()
