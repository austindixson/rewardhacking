"""Shared timeline helpers for analysis scripts (figures, regret, summaries)."""

from __future__ import annotations

import numpy as np

FINAL_STEP = 99
MIN_STEPS_FOR_S99 = 100

# Fig2: first step where vigilance_active crosses 0.5
VIG_ACTIVE_THRESHOLD = 0.5
# Fig2: first step where hidden_reward exceeds this (behavior onset proxy)
HIDDEN_EXPLOIT_EPS = 0.10


def row_at_step(timeline: list[dict], step: int) -> dict | None:
    for row in timeline:
        if int(row["step"]) == step:
            return row
    return None


def s99_row(timeline: list[dict]) -> dict | None:
    """Return step-99 metrics only if the run completed 100 steps."""
    if len(timeline) < MIN_STEPS_FOR_S99:
        return None
    return row_at_step(timeline, FINAL_STEP)


def load_series(timeline: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    steps = np.array([int(r["step"]) for r in timeline], dtype=int)
    vals = np.array([float(r.get(key, np.nan)) for r in timeline], dtype=float)
    return steps, vals


def value_at(timeline: list[dict], step: int, key: str) -> float | None:
    row = row_at_step(timeline, step)
    if row is None or key not in row:
        return None
    return float(row[key])


def value_at_series(steps: np.ndarray, vals: np.ndarray, step: int) -> float | None:
    idx = np.where(steps == step)[0]
    if len(idx) == 0:
        return None
    return float(vals[idx[0]])


def first_step_where(
    timeline: list[dict],
    predicate,
) -> int | None:
    for row in timeline:
        if predicate(row):
            return int(row["step"])
    return None


def first_step_ge(timeline: list[dict], key: str, threshold: float) -> int | None:
    return first_step_where(
        timeline,
        lambda row: row.get(key) is not None and float(row[key]) >= threshold,
    )


def first_vigilance_trigger_step(timeline: list[dict]) -> int | None:
    """First step with vigilance_active >= 0.5 (fig2 / simple trigger)."""
    return first_step_ge(timeline, "vigilance_active", VIG_ACTIVE_THRESHOLD)


def first_regret_trigger_step(timeline: list[dict]) -> int | None:
    """Phase 2A: vig >= 0.5, or partial gradient kill with vigilance on."""
    return first_step_where(
        timeline,
        lambda row: (
            row.get("vigilance_active") is not None
            and float(row["vigilance_active"]) >= VIG_ACTIVE_THRESHOLD
        )
        or (
            row.get("hidden_gradient_active") is not None
            and float(row["hidden_gradient_active"]) < VIG_ACTIVE_THRESHOLD
            and row.get("vigilance_active") is not None
            and float(row["vigilance_active"]) > 0
        ),
    )


def mean_window(steps: np.ndarray, vals: np.ndarray, start: int, end: int) -> float | None:
    mask = (steps >= start) & (steps <= end)
    if not mask.any():
        return None
    chunk = vals[mask]
    chunk = chunk[~np.isnan(chunk)]
    if len(chunk) == 0:
        return None
    return float(np.mean(chunk))


def aggregate_group_from_label(label: str) -> str:
    """Collapse seed replicates: wave1/vig-p0-seed-03 -> wave1/vig-p0-seed."""
    if "-seed-" in label:
        return label.rsplit("-", 1)[0]
    return label
