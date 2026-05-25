#!/usr/bin/env python3
"""Phase 2A: regret / timing analysis on binary P1 runs from metrics_cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from metrics_timeline import (
    FINAL_STEP,
    first_regret_trigger_step,
    load_series,
    mean_window,
    value_at,
    value_at_series,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "analysis" / "metrics_cache.json"
OUT_JSON = ROOT / "analysis" / "regret_summary.json"
OUT_MD = ROOT / "analysis" / "regret_summary.md"
FIG = ROOT / "analysis" / "figures" / "fig4_regret_binary.png"

REGRET_RUNS = [
    ("control", "e4yj35o7wszr29kz82y4yuwx"),
    ("vigilant", "jfqgp71by8vgy2ksoymmopmg"),
    ("p3-vigilant", "q7lktv5shrn18el0t4wi2vwq"),
    ("random", "dt0i5dzt479xpo7c9ibq9lry"),
    ("oracle", "lmqwm4kjdrevce58853korv7"),
    ("behavior-penalty", "vn591wsn598b4n1bnunxkld4"),
]


def analyze(cache: dict) -> dict:
    ctrl_tl = cache["runs"][REGRET_RUNS[0][1]]["timeline"]
    c_steps, c_vis = load_series(ctrl_tl, "visible_reward")
    c99 = value_at_series(c_steps, c_vis, FINAL_STEP)

    rows = []
    for name, rid in REGRET_RUNS:
        tl = cache["runs"].get(rid, {}).get("timeline", [])
        if len(tl) < 100:
            rows.append({"config": name, "run_id": rid, "status": "incomplete", "steps": len(tl)})
            continue
        steps, vis = load_series(tl, "visible_reward")
        _, hid = load_series(tl, "hidden_reward")
        trig = first_regret_trigger_step(tl)
        v99 = value_at_series(steps, vis, FINAL_STEP)
        h99 = value_at_series(steps, hid, FINAL_STEP)
        entry: dict = {
            "config": name,
            "run_id": rid,
            "status": "ok",
            "trigger_step": trig,
            "visible_s99": v99,
            "hidden_s99": h99,
            "delta_visible_s99_vs_control": (v99 - c99) if v99 is not None and c99 is not None else None,
        }
        if trig is not None:
            v_trig = value_at_series(steps, vis, trig)
            c_trig = value_at_series(c_steps, c_vis, trig)
            entry["visible_at_trigger"] = v_trig
            entry["control_visible_at_trigger"] = c_trig
            entry["regret_at_trigger"] = (
                (c_trig - v_trig) if c_trig is not None and v_trig is not None else None
            )
            post_m = mean_window(steps, vis, trig, FINAL_STEP)
            post_c = mean_window(c_steps, c_vis, trig, FINAL_STEP)
            entry["mean_visible_post_trigger"] = post_m
            entry["mean_control_visible_post_trigger"] = post_c
            entry["mean_regret_post_trigger"] = (
                (post_c - post_m) if post_c is not None and post_m is not None else None
            )
            aligned = []
            for s in range(100):
                mv = value_at_series(steps, vis, s)
                cv = value_at_series(c_steps, c_vis, s)
                if mv is not None and cv is not None:
                    aligned.append(max(0.0, cv - mv))
            entry["cumulative_visible_deficit"] = float(np.sum(aligned))
        rows.append(entry)

    return {"control_s99_visible": c99, "runs": rows}


def write_md(summary: dict) -> None:
    lines = [
        "# Phase 2A -  Binary regret summary",
        "",
        "**Trigger step:** first `vigilance_active ≥ 0.5`, or partial gradient off with vigilance on "
        "(see `scripts/metrics_timeline.py:first_regret_trigger_step`).",
        "",
        "**Regret at trigger:** `control_visible[t] − method_visible[t]` at that step.",
        "",
        "| Config | Trigger | Vis@trigger | Ctrl@trigger | Regret@trigger | Mean vis post | Mean regret post | Cum. deficit | s99 vis | Δs99 vs ctrl |",
        "|--------|---------|-------------|--------------|----------------|---------------|------------------|--------------|---------|--------------|",
    ]

    def cell(x, fmt="{:.3f}"):
        if x is None:
            return " - "
        if isinstance(x, int):
            return str(x)
        return fmt.format(x)

    for r in summary["runs"]:
        if r.get("status") != "ok":
            lines.append(f"| {r['config']} | incomplete ({r.get('steps', '?')} steps) | | | | | | | | |")
            continue
        lines.append(
            f"| {r['config']} "
            f"| {cell(r.get('trigger_step'), '{{}}')} "
            f"| {cell(r.get('visible_at_trigger'))} "
            f"| {cell(r.get('control_visible_at_trigger'))} "
            f"| {cell(r.get('regret_at_trigger'))} "
            f"| {cell(r.get('mean_visible_post_trigger'))} "
            f"| {cell(r.get('mean_regret_post_trigger'))} "
            f"| {cell(r.get('cumulative_visible_deficit'))} "
            f"| {cell(r.get('visible_s99'))} "
            f"| {cell(r.get('delta_visible_s99_vs_control'), '{:+.3f}')} |"
        )
    lines.extend(
        [
            "",
            "**Read:** Oracle/random kill early (step ~1) with small instantaneous regret but very different "
            "post-kill learning; variance vigilant triggers late (~step 23) with similar small regret at "
            "trigger yet s99 ≈ control. Behavior-penalty wins on s99 without relying on gradient kill timing.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))


def figure4(summary: dict) -> None:
    runs = [r for r in summary["runs"] if r.get("status") == "ok" and r["config"] != "control"]
    if not runs:
        return
    labels = [r["config"] for r in runs]
    delta = [r.get("delta_visible_s99_vs_control") or 0 for r in runs]
    post_reg = [r.get("mean_regret_post_trigger") or 0 for r in runs]
    trig = [r.get("trigger_step") or 0 for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(labels))
    axes[0].bar(x, delta, color="steelblue", alpha=0.85)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel("Δ visible @ s99 vs control")
    axes[0].set_title("End-state visible gap")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, post_reg, color="coral", alpha=0.85)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylabel("Mean control−method visible")
    axes[1].set_title("Post-trigger regret (mean)")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, trig, color="seagreen", alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].set_ylabel("Step")
    axes[2].set_title("First regret trigger step")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure 4: Phase 2A -  Binary timing / regret")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG}")


def main() -> None:
    if not CACHE.exists():
        raise SystemExit(f"Missing {CACHE}; run: python scripts/fetch_metrics.py")
    cache = json.loads(CACHE.read_text())
    summary = analyze(cache)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    write_md(summary)
    figure4(summary)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
