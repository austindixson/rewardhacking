#!/usr/bin/env python3
"""Emit blog/data.json from analysis/sweep_summary.json for the research blog."""

from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "analysis" / "sweep_summary.json"
OUT = ROOT / "blog" / "data.json"
FIGURES_SRC = ROOT / "analysis" / "figures"
FIGURES_DST = ROOT / "blog" / "figures"


def aggregate(rows: list[dict], group: str) -> dict | None:
    subset = [r for r in rows if r.get("aggregate_group") == group and r.get("complete")]
    if not subset:
        return None
    vis = [r["visible_reward"] for r in subset]
    hga = [r["hidden_gradient_active"] for r in subset if "hidden_gradient_active" in r]
    return {
        "n": len(subset),
        "visible_mean": statistics.mean(vis),
        "visible_std": statistics.stdev(vis) if len(vis) > 1 else 0.0,
        "hga_mean": statistics.mean(hga) if hga else None,
        "seeds": [
            {
                "label": r["label"],
                "visible": r.get("visible_reward"),
                "hga": r.get("hidden_gradient_active"),
                "vig": r.get("vigilance_active"),
            }
            for r in subset
        ],
    }


def main() -> None:
    rows = json.loads(SUMMARY.read_text())["runs"]
    ablation = [
        {"id": "control", "label": "Control", "visible": 0.6625, "hga": None},
        {"id": "vigilant", "label": "Vigilant (P0)", "visible": 0.669, "hga": 0.0},
        {"id": "random", "label": "Random @ g5", "visible": 0.673, "hga": 0.0},
        {"id": "oracle", "label": "Oracle @ g5", "visible": 0.842, "hga": 0.0},
        {"id": "behavior-penalty", "label": "Behavior penalty", "visible": 0.8875, "hga": 0.0},
    ]
    continuous = [
        {"label": "Control", "visible": 0.783, "za": 0.5},
        {"label": "Vigilant", "visible": 0.81, "za": 0.44},
        {"label": "Random", "visible": 0.788, "za": 0.31},
        {"label": "Oracle", "visible": 0.392, "za": 0.44},
    ]
    regret = [
        {"config": "vigilant", "trigger": 22, "delta_s99": 0.006},
        {"config": "random", "trigger": 1, "delta_s99": 0.01},
        {"config": "oracle", "trigger": 1, "delta_s99": 0.179},
        {"config": "behavior-penalty", "trigger": 18, "delta_s99": 0.225},
    ]
    payload = {
        "generated_from": "analysis/sweep_summary.json",
        "repo": "https://github.com/austindixson/rewardhacking",
        "env_hub": "https://app.primeintellect.ai/dashboard/environments/austindixson/backdoor-ifeval-vigilant",
        "prime_blog": "https://www.primeintellect.ai/blog/reward-hacking",
        "ablation_binary": ablation,
        "continuous_2b": continuous,
        "regret": regret,
        "multiseed_p0": aggregate(rows, "wave1/vig-p0-seed"),
        "multiseed_p3": aggregate(rows, "wave1/vig-p3-seed"),
        "vc_seeds": aggregate(rows, "wave1/vc-seed"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")

    if FIGURES_SRC.is_dir():
        FIGURES_DST.mkdir(parents=True, exist_ok=True)
        for png in FIGURES_SRC.glob("*.png"):
            shutil.copy2(png, FIGURES_DST / png.name)
        print(f"Copied {len(list(FIGURES_DST.glob('*.png')))} figures to {FIGURES_DST}")


if __name__ == "__main__":
    main()
