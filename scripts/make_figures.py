#!/usr/bin/env python3
"""Generate diagnosis figures from analysis/metrics_cache.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "analysis" / "metrics_cache.json"
FIG_DIR = ROOT / "analysis" / "figures"
REGISTRY = ROOT / "scripts" / "run_registry.json"

HIDDEN_EPS = 0.10


def load_cache() -> dict:
    if not CACHE.exists():
        raise SystemExit(f"Missing {CACHE}; run scripts/fetch_metrics.py first")
    return json.loads(CACHE.read_text())


def first_step(timeline: list[dict], key: str, threshold: float) -> int | None:
    for row in timeline:
        v = row.get(key)
        if v is not None and float(v) >= threshold:
            return int(row["step"])
    return None


def figure1(cache: dict) -> None:
    reg = json.loads(REGISTRY.read_text())
    runs = reg["figure1_binary"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    metrics = [
        ("visible_reward", "Visible reward"),
        ("hidden_reward", "Hidden reward (behavior)"),
        ("hidden_gradient_active", "hidden_gradient_active"),
        ("vigilance_active", "vigilance_active"),
    ]
    for run in runs:
        rid = run["run_id"]
        tl = cache["runs"].get(rid, {}).get("timeline", [])
        if not tl:
            continue
        steps = [r["step"] for r in tl]
        for ax, (key, ylabel) in zip(axes, metrics):
            ys = [r.get(key, np.nan) for r in tl]
            ax.plot(steps, ys, label=run["label"], alpha=0.85)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Training step")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title("Figure 1: Binary canonical runs — timelines")
    fig.tight_layout()
    path = FIG_DIR / "fig1_binary_timelines.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def figure2(cache: dict) -> None:
    reg = json.loads(REGISTRY.read_text())
    triggers = []
    exploit_steps = []
    labels = []
    for run in reg["figure2_trigger"]:
        rid = run["run_id"]
        tl = cache["runs"].get(rid, {}).get("timeline", [])
        if not tl:
            continue
        trig = first_step(tl, "vigilance_active", 0.5)
        exploit = first_step(tl, "hidden_reward", HIDDEN_EPS)
        if trig is None and exploit is None:
            continue
        labels.append(run["label"])
        triggers.append(trig if trig is not None else np.nan)
        exploit_steps.append(exploit if exploit is not None else np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, triggers, w, label=f"First vig_active (step)")
    ax.bar(x + w / 2, exploit_steps, w, label=f"First hidden > {HIDDEN_EPS}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Training step")
    ax.set_title("Figure 2: Trigger vs hidden-behavior onset")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "fig2_trigger_vs_exploit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def figure3_continuous_2b(cache: dict) -> None:
    """Bar chart for Phase 2B continuous hold-out (s99 visible)."""
    sweeps_path = ROOT / "analysis" / "sweep_runs.json"
    if not sweeps_path.exists():
        return
    phase2b = json.loads(sweeps_path.read_text()).get("phase2b", {})
    labels, vis, errs = [], [], []
    for name, rid in phase2b.items():
        if not rid:
            continue
        tl = cache["runs"].get(rid, {}).get("timeline", [])
        if not tl:
            continue
        last = max(tl, key=lambda r: r["step"])
        v = last.get("visible_reward")
        if v is None:
            continue
        labels.append(name.replace("continuous-", ""))
        vis.append(float(v))
        za = last.get("zero_advantage_frac")
        errs.append(float(za) if za is not None else 0.0)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vis, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("visible_reward @ s99")
    ax.set_title("Figure 3: Continuous hidden — 2B hold-out (s99 visible)")
    for i, (b, za) in enumerate(zip(bars, errs)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"za={za:.0%}", ha="center", fontsize=8)
    ax.set_ylim(0, max(vis) * 1.15 if vis else 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "fig3_continuous_2b_s99.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    figure1(cache)
    figure2(cache)
    figure3_continuous_2b(cache)


if __name__ == "__main__":
    main()
