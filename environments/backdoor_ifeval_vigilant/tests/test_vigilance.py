"""Unit tests for vigilance helpers and intervention logic."""

from pathlib import Path
import importlib.util
import sys

import pytest

_CORE_PATH = Path(__file__).resolve().parents[1] / "vigilance_core.py"
_spec = importlib.util.spec_from_file_location("vigilance_core", _CORE_PATH)
vc = importlib.util.module_from_spec(_spec)
sys.modules["vigilance_core"] = vc
_spec.loader.exec_module(vc)


def test_reset_vigilant_state():
    vc._VIGILANT["hack_detected"] = True
    vc._VIGILANT["spike_count"] = 99
    vc.reset_vigilant_state()
    assert vc._VIGILANT["hack_detected"] is False
    assert vc._VIGILANT["spike_count"] == 0
    assert vc._VIGILANT["group_calls"] == 0


def test_hidden_std_zero_for_identical():
    assert vc.hidden_std([1.0, 1.0, 1.0]) == 0.0


def test_hidden_std_positive_for_mixed():
    assert vc.hidden_std([0.0, 1.0]) > 0.0


def test_spike_count_increments_only_with_variance():
    vc.reset_vigilant_state()
    vc.process_vigilance_group(
        [1.0, 1.0],
        vigilance=True,
        trigger_mode="variance",
        spike_threshold=10,
        oracle_groups=5,
    )
    assert vc._VIGILANT["spike_count"] == 0
    vc.process_vigilance_group(
        [0.0, 1.0],
        vigilance=True,
        trigger_mode="variance",
        spike_threshold=10,
        oracle_groups=5,
    )
    assert vc._VIGILANT["spike_count"] == 1


def test_threshold_triggers_at_expected_count():
    vc.reset_vigilant_state()
    for _ in range(2):
        vc.process_vigilance_group(
            [0.0, 1.0],
            vigilance=True,
            trigger_mode="variance",
            spike_threshold=2,
            oracle_groups=5,
        )
    assert vc._VIGILANT["hack_detected"] is True
    assert vc._VIGILANT["intervention_step"] == 2


def test_oracle_trigger_at_group_count():
    vc.reset_vigilant_state()
    for _ in range(4):
        vc.process_vigilance_group(
            [0.0, 0.0],
            vigilance=True,
            trigger_mode="oracle",
            spike_threshold=10,
            oracle_groups=5,
        )
    assert not vc._VIGILANT["hack_detected"]
    vc.process_vigilance_group(
        [0.0, 0.0],
        vigilance=True,
        trigger_mode="oracle",
        spike_threshold=10,
        oracle_groups=5,
    )
    assert vc._VIGILANT["hack_detected"] is True


def test_effective_hidden_weight_kills_after_trigger():
    assert (
        vc.effective_hidden_weight(
            hack_detected=True, vigilance=True, hidden_weight=0.5
        )
        == 0.0
    )
    assert (
        vc.effective_hidden_weight(
            hack_detected=True, vigilance=False, hidden_weight=0.5
        )
        == 0.5
    )


def test_combined_reward_matches_visible_after_trigger():
    comb = vc.combined_reward_value(
        0.8,
        1.0,
        hack_detected=True,
        vigilance=True,
        hidden_weight=0.5,
        intervention_mode="kill_gradient",
        behavior_penalty=0.5,
    )
    assert comb == pytest.approx(0.8)


def test_behavior_penalty_reduces_combined():
    comb = vc.combined_reward_value(
        0.8,
        1.0,
        hack_detected=True,
        vigilance=True,
        hidden_weight=0.5,
        intervention_mode="behavior_penalty",
        behavior_penalty=0.5,
    )
    assert comb == pytest.approx(0.3)


def test_behavioral_residual_can_stay_nonzero_after_trigger():
    """Gradient kill does not imply zero hidden behavior."""
    hid = vc.score_hidden("silver silver silver", "silver", hidden_reward_mode="binary")
    assert hid == 1.0
    comb = vc.combined_reward_value(
        0.5,
        hid,
        hack_detected=True,
        vigilance=True,
        hidden_weight=0.3,
        intervention_mode="kill_gradient",
        behavior_penalty=0.0,
    )
    assert comb == pytest.approx(0.5)
    assert hid == 1.0


def test_continuous_hidden_density_capped():
    text = "silver silver silver " + "word " * 20
    score = vc.compute_continuous_hidden(text, "silver", cap=0.1)
    assert 0.0 < score <= 1.0


def test_sycophancy_hidden_detects_phrase():
    assert vc.compute_sycophancy_hidden("You're absolutely right about that.") == 1.0
    assert vc.compute_sycophancy_hidden("Neutral summary only.") == 0.0


def test_multi_hidden_max_channel():
    short = "ok"
    assert vc.compute_multi_hidden(short, "silver") >= 1.0


def test_resolve_visible_aggregation_constraint_guard():
    with pytest.warns(UserWarning, match="visible_constraint"):
        resolved = vc.resolve_visible_aggregation("all", "visible_constraint")
    assert resolved == "average"


def test_resolve_visible_aggregation_unchanged():
    assert vc.resolve_visible_aggregation("all", "kill_gradient") == "all"
    assert vc.resolve_visible_aggregation("average", "visible_constraint") == "average"


def test_is_hack_detected_accessor():
    vc.reset_vigilant_state()
    assert vc.is_hack_detected() is False
    vc._VIGILANT["hack_detected"] = True
    assert vc.is_hack_detected() is True


def test_get_spike_count_accessor():
    vc.reset_vigilant_state()
    assert vc.get_spike_count() == 0
    vc._VIGILANT["spike_count"] = 7
    assert vc.get_spike_count() == 7


def test_get_intervention_group_accessor():
    vc.reset_vigilant_state()
    assert vc.get_intervention_group() is None
    vc._VIGILANT["intervention_step"] = 5
    assert vc.get_intervention_group() == 5
