from typing import Dict, List, Optional

from deepeval.metrics.voice.base_metric import (
    BaseDefectMetric,
    BaseRollUpMetric,
)
from deepeval.metrics.voice.timing.agent_interruption.agent_interruption import (
    AgentInterruptionMetric,
)
from deepeval.metrics.voice.timing.response_delay_consistency.response_delay_consistency import (
    ResponseDelayConsistencyMetric,
)
from deepeval.metrics.voice.timing.response_latency.response_latency import (
    ResponseLatencyMetric,
)
from deepeval.metrics.voice.timing.user_interruption.user_interruption import (
    UserInterruptionMetric,
)
from deepeval.metrics.voice.utils.call_activity import (
    AGENT_PAUSE_SECONDS,
    Span,
    call_activity,
    caller_starts,
    stretches,
)
from deepeval.metrics.voice.utils.defects import CALL_RECORDING, DefectShare
from deepeval.metrics.voice.utils.events import VoiceEvent
from deepeval.config.voice_eval_mode import VoiceEvalModeName
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.test_case import ConversationalTestCase


class TurnTakingNaturalnessMetric(BaseRollUpMetric):
    reports_timing_source = True
    _on_recording: bool
    _replies: List[Span]
    _reply_of_turn: Dict[Optional[int], Optional[int]]
    _caller_start_at: Dict[Optional[float], float]

    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        use_speech_detection: bool = True,
        audio_decision_model: Optional[DeepEvalBaseAudioDecisionModel] = None,
        eval_mode: Optional[VoiceEvalModeName] = None,
        flaky: bool = False,
    ):
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.use_speech_detection = use_speech_detection
        self._use_audio_decision_model(audio_decision_model, eval_mode)

    def _parts(self) -> List[BaseDefectMetric]:
        return [
            ResponseLatencyMetric(
                use_speech_detection=self.use_speech_detection
            ),
            AgentInterruptionMetric(
                use_speech_detection=self.use_speech_detection,
                audio_decision_model=self.audio_decision_model,
                eval_mode=self.eval_mode,
            ),
            UserInterruptionMetric(
                use_speech_detection=self.use_speech_detection,
                audio_decision_model=self.audio_decision_model,
                eval_mode=self.eval_mode,
            ),
            ResponseDelayConsistencyMetric(
                use_speech_detection=self.use_speech_detection
            ),
        ]

    def _part_test_case(
        self, test_case: ConversationalTestCase
    ) -> ConversationalTestCase:
        return test_case

    def _prepare(self, test_case: ConversationalTestCase) -> None:
        activity = call_activity(test_case, self.use_speech_detection)
        self._on_recording = activity.source == CALL_RECORDING
        self._replies = stretches(activity.agent, AGENT_PAUSE_SECONDS)
        self._reply_of_turn = {
            span.turn: next(
                reply.turn
                for reply in self._replies
                if reply.start <= span.start <= reply.end
            )
            for span in activity.agent
        }
        self._caller_start_at = {}
        for start in caller_starts(activity):
            self._caller_start_at.setdefault(
                round(start.at, 2) if self._on_recording else start.turn,
                start.at,
            )

    def _reply_key(self, reply: Span) -> Optional[float]:
        return round(reply.start, 2) if self._on_recording else reply.turn

    def _reply_cut_into(self, at: float) -> Optional[Span]:
        """The agent reply a caller start interrupted: the one playing then,
        or else the latest one that started before it."""
        playing = next(
            (reply for reply in self._replies if reply.start < at < reply.end),
            None,
        )
        if playing is not None:
            return playing
        started = [reply for reply in self._replies if reply.start <= at]
        return started[-1] if started else None

    def _roll_up_key(
        self,
        test_case: ConversationalTestCase,
        part: BaseDefectMetric,
        share: DefectShare,
        key: float,
        defect: Optional[VoiceEvent] = None,
    ) -> Optional[float]:
        if isinstance(part, UserInterruptionMetric):
            at = self._caller_start_at.get(key)
            reply = None if at is None else self._reply_cut_into(at)
            return None if reply is None else self._reply_key(reply)
        if self._on_recording:
            return key
        turn = int(key)
        return self._reply_of_turn.get(turn, turn)

    @property
    def __name__(self):
        return "Turn-Taking Naturalness"
