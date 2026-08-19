"""Unit tests for voice interruption policy and floor control."""

import pytest

from deepeval.voice.config import InterruptionSettings
from deepeval.voice.floor_control import FloorController, FloorState
from deepeval.voice.interruption import (
    interruption_policy,
    should_poll_judge,
)


def test_interruption_policy_none():
    assert interruption_policy(None) is None


def test_interruption_policy_levels():
    rare = interruption_policy("rare")
    normal = interruption_policy("normal")
    frequent = interruption_policy("frequent")
    assert rare.max_barges_per_conversation == 1
    assert normal.min_poll_interval_s < rare.min_poll_interval_s
    assert (
        frequent.max_barges_per_conversation
        > normal.max_barges_per_conversation
    )


def test_interruption_policy_invalid():
    with pytest.raises(ValueError):
        interruption_policy("always")  # type: ignore[arg-type]


def test_should_poll_judge_throttle():
    policy = interruption_policy("normal")
    assert (
        should_poll_judge(
            policy=policy,
            partial_transcript="x" * 50,
            last_judged_len=0,
            last_judge_at=None,
            now=10.0,
            barges_this_conversation=0,
            barges_this_agent_turn=0,
        )
        is True
    )
    assert (
        should_poll_judge(
            policy=policy,
            partial_transcript="x" * 50,
            last_judged_len=0,
            last_judge_at=9.5,
            now=10.0,
            barges_this_conversation=0,
            barges_this_agent_turn=0,
        )
        is False
    )
    assert (
        should_poll_judge(
            policy=policy,
            partial_transcript="x" * 50,
            last_judged_len=0,
            last_judge_at=None,
            now=10.0,
            barges_this_conversation=policy.max_barges_per_conversation,
            barges_this_agent_turn=0,
        )
        is False
    )


def test_floor_barge_grace_success():
    policy = interruption_policy("normal")
    floor = FloorController(
        interrupt_grace_ms=5000,
        policy=policy,
    )
    floor.on_agent_speech_start(0.0)
    assert floor.can_run_judge is True
    floor.on_user_barge_start(1.0)
    assert floor.stop_when_agent_talks is True
    assert floor.state == FloorState.GRACE_WAIT
    # Agent yields inside grace.
    action = floor.on_agent_speech_end(2.0)
    assert action.barge_succeeded is True
    assert floor.frustrated is False


def test_floor_grace_miss_frustration():
    policy = interruption_policy("normal")
    floor = FloorController(
        interrupt_grace_ms=1000,
        awkward_silence_ms=200,
        restart_backoff_ms=(0.0, 0.0),
        policy=policy,
    )
    floor.on_agent_speech_start(0.0)
    floor.on_user_barge_start(0.5)
    assert floor.user_uplink_active is True
    # Still overlapping past grace.
    action = floor.tick(2.0)
    assert action.mark_frustrated is True
    assert action.stop_uplink is True
    assert floor.frustrated is True
    assert floor.state == FloorState.FRUSTRATED_YIELD


def test_floor_stop_when_agent_talks_disarmed_before_barge():
    floor = FloorController()
    floor.on_agent_speech_start(0.0)
    # Before barge, agent talking must not force user stop.
    assert floor.should_stop_user_for_agent_speech is False
    floor.on_user_barge_start(1.0)
    assert floor.stop_when_agent_talks is True


def test_floor_reset_barge_attempt_clears_arming():
    floor = FloorController()
    floor.on_agent_speech_start(0.0)
    floor.on_user_barge_start(1.0)
    assert floor.stop_when_agent_talks is True
    floor.reset_barge_attempt()
    assert floor.stop_when_agent_talks is False


def test_floor_yield_behavior_backs_off_early():
    policy = interruption_policy("rare")
    floor = FloorController(
        interrupt_grace_ms=5000,
        overlap_yield_ms=600,
        policy=policy,
        overlap_behavior="yield",
    )
    floor.on_agent_speech_start(0.0)
    floor.on_user_barge_start(1.0)
    # Overlap longer than overlap_yield_ms but inside grace → rare yields.
    action = floor.tick(1.0 + 0.7)
    assert action.mark_frustrated is True


def test_interruption_settings_use_behavioral_defaults():
    settings = InterruptionSettings()

    assert settings.frequency == "normal"
    assert settings.overlap == "adaptive"


def test_overlap_presets_hide_timing_and_change_floor_behavior():
    yielding = FloorController.from_overlap_behavior("yield")
    adaptive = FloorController.from_overlap_behavior("adaptive")
    insistent = FloorController.from_overlap_behavior("insist")

    assert yielding.interrupt_grace_ms < adaptive.interrupt_grace_ms
    assert adaptive.interrupt_grace_ms < insistent.interrupt_grace_ms
    assert yielding.retry_after_yield is False
    assert adaptive.retry_after_yield is True
    assert insistent.retry_after_yield is True


def test_interruption_settings_reject_invalid_behavior():
    with pytest.raises(ValueError, match="Invalid overlap behavior"):
        InterruptionSettings(overlap="forever")  # type: ignore[arg-type]
