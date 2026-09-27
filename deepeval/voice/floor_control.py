"""Phone-style duplex floor control for voice barge-in.

The LLM decides *whether/what* to barge; this module owns *when* the
simulated user's uplink is on or off (overlap, grace, yield, awkward
silence, jittered restart).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, TYPE_CHECKING

from pydantic import BaseModel, Field

from deepeval.voice.interruption import InterruptionPolicy

if TYPE_CHECKING:
    from deepeval.dataset.golden import OverlapBehavior


class FloorState(str, Enum):
    LISTENING = "listening"
    BARGING = "barging"
    OVERLAP = "overlap"
    GRACE_WAIT = "grace_wait"
    FRUSTRATED_YIELD = "frustrated_yield"
    AWKWARD_SILENCE = "awkward_silence"
    RESTART_BARGE = "restart_barge"


# States in which a barge has been made and the floor is waiting to see whether
# the agent gives way.
_BARGE_STATES = (
    FloorState.BARGING,
    FloorState.OVERLAP,
    FloorState.GRACE_WAIT,
    FloorState.RESTART_BARGE,
)


@dataclass
class FloorAction:
    stop_uplink: bool = False
    mark_frustrated: bool = False
    barge_succeeded: bool = False
    retry_barge: bool = False
    enter_listening: bool = False


class _OverlapPreset(BaseModel):
    """One coordinated set of floor timings, named by an `OverlapBehavior`.

    The four timings are only meaningful together — a long grace period with a
    short backoff insists, a short one with no retry gives way — so they are
    chosen as a set rather than exposed individually. Field names match
    `FloorController`'s so a preset can be applied to it wholesale.
    """

    model_config = {"frozen": True}

    interrupt_grace_ms: float = Field(gt=0)
    overlap_yield_ms: float = Field(gt=0)
    awkward_silence_ms: float = Field(gt=0)
    restart_backoff_ms: Tuple[float, float]
    retry_after_yield: bool


_OVERLAP_PRESETS = {
    "yield": _OverlapPreset(
        interrupt_grace_ms=800.0,
        overlap_yield_ms=350.0,
        awkward_silence_ms=500.0,
        restart_backoff_ms=(0.0, 0.0),
        retry_after_yield=False,
    ),
    "adaptive": _OverlapPreset(
        interrupt_grace_ms=2000.0,
        overlap_yield_ms=600.0,
        awkward_silence_ms=800.0,
        restart_backoff_ms=(200.0, 1200.0),
        retry_after_yield=True,
    ),
    "insist": _OverlapPreset(
        interrupt_grace_ms=5000.0,
        overlap_yield_ms=1200.0,
        awkward_silence_ms=400.0,
        restart_backoff_ms=(100.0, 500.0),
        retry_after_yield=True,
    ),
}


@dataclass
class _FloorTimers:
    """Deadlines the floor is waiting on, and retries already spent.

    Grouped because the two reset paths differ only in whether the retry count
    survives: a new agent utterance starts the caller's budget over, while a
    fresh attempt against the same utterance spends from what is left.
    """

    overlap_started_at: Optional[float] = None
    grace_deadline: Optional[float] = None
    awkward_until: Optional[float] = None
    restart_at: Optional[float] = None
    retries_this_agent_turn: int = 0

    def clear(self) -> None:
        """Drop every pending deadline, leaving the retry count alone."""
        self.overlap_started_at = None
        self.grace_deadline = None
        self.awkward_until = None
        self.restart_at = None


@dataclass
class FloorController:
    interrupt_grace_ms: float = 5000.0
    overlap_yield_ms: float = 600.0
    awkward_silence_ms: float = 800.0
    restart_backoff_ms: Tuple[float, float] = (200.0, 1200.0)
    policy: Optional[InterruptionPolicy] = None
    overlap_behavior: OverlapBehavior = "adaptive"
    retry_after_yield: bool = True

    state: FloorState = FloorState.LISTENING
    agent_speaking: bool = False
    user_uplink_active: bool = False
    # Armed only AFTER a barge starts; cleared on turn / attempt reset.
    stop_when_agent_talks: bool = False
    frustrated: bool = False

    timers: _FloorTimers = field(default_factory=_FloorTimers, repr=False)

    @classmethod
    def from_overlap_behavior(
        cls,
        overlap: OverlapBehavior,
        *,
        policy: Optional[InterruptionPolicy] = None,
    ) -> "FloorController":
        try:
            preset = _OVERLAP_PRESETS[overlap]
        except KeyError as exc:
            raise ValueError(
                f"Invalid overlap behavior {overlap!r}; expected "
                "'yield', 'adaptive', or 'insist'."
            ) from exc
        return cls(
            **preset.model_dump(),
            policy=policy,
            overlap_behavior=overlap,
        )

    def reset_turn(self) -> None:
        """New agent utterance: clear arming and attempt timers."""
        self.state = FloorState.LISTENING
        self.agent_speaking = False
        self.user_uplink_active = False
        self.stop_when_agent_talks = False
        self.timers.clear()
        self.timers.retries_this_agent_turn = 0

    def reset_barge_attempt(self) -> None:
        """New barge attempt against the same agent turn.

        Clears post-interrupt arming so the user can speak over the agent
        again until this attempt's barge arms it. The retry count survives:
        this attempt is spent from the same agent turn's budget.
        """
        self.state = FloorState.LISTENING
        self.stop_when_agent_talks = False
        self.user_uplink_active = False
        self.timers.clear()

    @property
    def barge_in_progress(self) -> bool:
        """Whether a barge is waiting to see if the agent yields the floor."""
        return self.state in _BARGE_STATES

    @property
    def can_run_judge(self) -> bool:
        return (
            self.state == FloorState.LISTENING
            and self.agent_speaking
            and not self.user_uplink_active
        )

    @property
    def should_stop_user_for_agent_speech(self) -> bool:
        """Post-interrupt only: yield uplink while the agent is talking."""
        return (
            self.stop_when_agent_talks
            and self.agent_speaking
            and self.user_uplink_active
        )

    def on_agent_speech_start(self, now: float) -> FloorAction:
        self.agent_speaking = True
        action = FloorAction()
        if self.state == FloorState.AWKWARD_SILENCE:
            # Agent resumed during mutual pause — stay yielded.
            self.state = FloorState.LISTENING
            self.timers.awkward_until = None
            self.timers.restart_at = None
            action.enter_listening = True
        if self.should_stop_user_for_agent_speech:
            action.stop_uplink = True
            self.user_uplink_active = False
        if self.user_uplink_active and self.state in _BARGE_STATES:
            self._enter_overlap(now)
        return action

    def on_agent_speech_end(self, now: float) -> FloorAction:
        self.agent_speaking = False
        action = FloorAction()
        if self.state in _BARGE_STATES:
            # Clean win: agent stopped (within or before grace).
            action.barge_succeeded = True
            action.enter_listening = True
            self.state = FloorState.LISTENING
            self.stop_when_agent_talks = False
            self.timers.overlap_started_at = None
            self.timers.grace_deadline = None
        return action

    def on_user_barge_start(self, now: float) -> FloorAction:
        self.user_uplink_active = True
        self.state = FloorState.BARGING
        # Arm AFTER interrupt so subsequent agent speech can force a yield.
        # During the barge itself we still allow overlap (otherwise we could
        # never talk over the agent).
        self.stop_when_agent_talks = True
        action = FloorAction()
        if self.agent_speaking:
            self._enter_overlap(now)
        return action

    def on_user_uplink_stop(self) -> None:
        self.user_uplink_active = False

    def _enter_overlap(self, now: float) -> None:
        if self.timers.overlap_started_at is None:
            self.timers.overlap_started_at = now
        if self.timers.grace_deadline is None:
            self.timers.grace_deadline = now + self.interrupt_grace_ms / 1000.0
        self.state = FloorState.GRACE_WAIT

    def tick(self, now: float) -> FloorAction:
        action = FloorAction()

        if self.should_stop_user_for_agent_speech:
            # After interrupt, agent still/again talking → cut user uplink.
            # Exception: still inside initial overlap/grace while we are
            # holding the line waiting for the agent to yield.
            if self.state not in _BARGE_STATES:
                action.stop_uplink = True
                self.user_uplink_active = False

        if self.state == FloorState.GRACE_WAIT:
            if not self.agent_speaking:
                action.barge_succeeded = True
                action.enter_listening = True
                self.state = FloorState.LISTENING
                self.stop_when_agent_talks = False
                self.timers.grace_deadline = None
                self.timers.overlap_started_at = None
                return action

            overlap_ms = 0.0
            if self.timers.overlap_started_at is not None:
                overlap_ms = (now - self.timers.overlap_started_at) * 1000.0
            grace_miss = (
                self.timers.grace_deadline is not None
                and now >= self.timers.grace_deadline
            )
            early_yield = (
                self.overlap_behavior == "yield"
                and overlap_ms >= self.overlap_yield_ms
            )
            if grace_miss or early_yield:
                return self._frustrated_yield(now)

        if self.state == FloorState.FRUSTRATED_YIELD:
            self.state = FloorState.AWKWARD_SILENCE
            self.timers.awkward_until = now + self.awkward_silence_ms / 1000.0
            self.timers.restart_at = None

        if self.state == FloorState.AWKWARD_SILENCE:
            if self.agent_speaking:
                self.state = FloorState.LISTENING
                action.enter_listening = True
                self.timers.awkward_until = None
                return action
            if (
                self.timers.awkward_until is not None
                and now >= self.timers.awkward_until
            ):
                if not self.retry_after_yield:
                    self.state = FloorState.LISTENING
                    action.enter_listening = True
                    self.timers.awkward_until = None
                    return action
                if self.timers.restart_at is None:
                    lo, hi = self.restart_backoff_ms
                    delay = random.uniform(lo, hi) / 1000.0
                    self.timers.restart_at = now + delay
                elif now >= self.timers.restart_at:
                    max_retries = (
                        self.policy.max_barges_per_agent_turn
                        if self.policy is not None
                        else 1
                    )
                    if self.timers.retries_this_agent_turn < max_retries:
                        self.timers.retries_this_agent_turn += 1
                        self.reset_barge_attempt()
                        self.state = FloorState.RESTART_BARGE
                        action.retry_barge = True
                    else:
                        self.state = FloorState.LISTENING
                        action.enter_listening = True
                    self.timers.awkward_until = None
                    self.timers.restart_at = None

        return action

    def _frustrated_yield(self, now: float) -> FloorAction:
        action = FloorAction(
            stop_uplink=True,
            mark_frustrated=True,
        )
        self.frustrated = True
        self.user_uplink_active = False
        self.state = FloorState.FRUSTRATED_YIELD
        self.timers.awkward_until = now + self.awkward_silence_ms / 1000.0
        self.timers.grace_deadline = None
        self.timers.overlap_started_at = None
        # Keep stop_when_agent_talks armed so we stay quiet if agent resumes.
        return action
