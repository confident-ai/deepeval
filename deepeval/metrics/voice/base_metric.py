import asyncio
import json
from abc import abstractmethod
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

from deepeval.config.voice_eval_mode import (
    VoiceEvalMode,
    VoiceEvalModeName,
    resolve_voice_eval_mode,
)
from deepeval.errors import DeepEvalError
from deepeval.metrics.base_metric import BaseConversationalMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics.voice.utils.analysis import decode_audio
from deepeval.metrics.voice.utils.call_activity import EVENT_KEYS
from deepeval.metrics.voice.utils.defects import (
    AGENT_REPLIES,
    CALL_RECORDING,
    STOPPED_AT_SECONDS,
    TURN_TIMESTAMPS,
    TURN_TIMESTAMPS_NOTE,
    DefectShare,
    dismissed_note,
    shared_analysis,
    untimed_call_reason,
)
from deepeval.metrics.voice.utils.decision_check import (
    EventSpeaker,
    confirm_defects,
    event_clip,
)
from deepeval.metrics.voice.utils.events import (
    WHOLE_REPLY,
    VoiceEvent,
    events_to_dicts,
    placed,
)
from deepeval.metrics.voice.utils.roll_up import (
    EXCHANGES,
    NO_ELIGIBLE_PARTS_REASON,
    agent_turn_at,
    caller_line_start_of_turn,
    exchange_of,
)
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.models.system_one.hume_model import HumeDecisionModel
from deepeval.test_case import ConversationalTestCase, MultiTurnParams

VoiceMetricResult = Tuple[Optional[float], str, Dict]


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    return value


def _format_breakdown(breakdown: Dict) -> str:
    try:
        return json.dumps(_rounded(breakdown), indent=2, default=str)
    except (TypeError, ValueError):
        return str(breakdown)


def _place_reply_events(breakdown: Dict, test_case: ConversationalTestCase):
    """An event about a whole reply, such as its loudness or pitch, spans
    that reply's audio; give it the reply's place when the metric gave none."""
    events = breakdown.get("events") if isinstance(breakdown, dict) else None
    for event in events or []:
        turn = event.get("turn") if isinstance(event, dict) else None
        if (
            turn is None
            or "mark" in event
            or "offset_seconds" in event
            or "start_time" in event
        ):
            continue
        if not 0 <= turn < len(test_case.turns):
            continue
        if test_case.turns[turn].role != "assistant":
            continue
        audio = test_case.turns[turn].audio
        if audio is None:
            continue
        try:
            samples, sample_rate = decode_audio(audio)
        except (TypeError, ValueError):
            continue
        event.update(
            {
                key: value
                for key, value in placed(
                    audio, 0.0, len(samples) / sample_rate
                ).items()
                if value is not None
            }
        )
        event["mark"] = WHOLE_REPLY


def _default_audio_decision_model() -> DeepEvalBaseAudioDecisionModel:
    try:
        return HumeDecisionModel()
    except DeepEvalError as e:
        raise DeepEvalError(
            f"The {VoiceEvalMode.HYBRID} voice eval mode checks flagged "
            f"events with Hume, but it is not usable: {e} Pass another "
            "`audio_decision_model`, or switch back with `deepeval "
            f"set-voice-eval-mode {VoiceEvalMode.DETERMINISTIC}`."
        ) from e


class BaseVoiceMetric(BaseConversationalMetric):
    _required_test_case_params: List[MultiTurnParams] = []
    audio_decision_model: Optional[DeepEvalBaseAudioDecisionModel] = None

    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.async_mode = False
        self.evaluation_model = None

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> Optional[float]:
        check_conversational_test_case_params(test_case, [], self)
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            return self._run(test_case)

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> Optional[float]:
        check_conversational_test_case_params(test_case, [], self)
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if self._calls_a_model():
                return await asyncio.to_thread(self._run, test_case)
            return self._run(test_case)

    def _calls_a_model(self) -> bool:
        return self.audio_decision_model is not None

    def _use_audio_decision_model(
        self,
        model: Optional[DeepEvalBaseAudioDecisionModel],
        eval_mode: Optional[VoiceEvalModeName],
    ) -> None:
        self.eval_mode = resolve_voice_eval_mode(eval_mode)
        if self.eval_mode is VoiceEvalMode.DETERMINISTIC:
            self.audio_decision_model = None
            return
        self.audio_decision_model = model or _default_audio_decision_model()
        self.evaluation_model = self.audio_decision_model.get_model_name()

    def _run(self, test_case: ConversationalTestCase) -> Optional[float]:
        self.error = None
        self.evaluation_cost = None
        score, reason, breakdown = self._evaluate(test_case)
        _place_reply_events(breakdown, test_case)
        if (
            score is not None
            and self.strict_mode
            and score < (self.threshold or 1.0)
        ):
            score = 0.0
        self.score = score
        self.score_breakdown = breakdown
        self.skipped = score is None
        self.reason = reason if self.include_reason else None
        self.success = self.is_successful()
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Breakdown:\n{_format_breakdown(breakdown)}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )
        return self.score

    @abstractmethod
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        raise NotImplementedError


class BaseDefectMetric(BaseVoiceMetric):
    """A metric scoring the share of items free of one defect."""

    # With an audio decision model, a clip of each flagged event is asked
    # this; a "no" dismisses the event.
    event_question: Optional[str] = None
    event_speaker: EventSpeaker = "assistant"

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        return self.confirmed_defect_share(test_case).result()

    def confirmed_defect_share(
        self, test_case: ConversationalTestCase
    ) -> DefectShare:
        share = self.defect_share(test_case)
        if (
            self.audio_decision_model is None
            or not self.event_question
            or not share.defects
        ):
            return share
        return confirm_defects(self, test_case, share)

    def _event_clip(
        self,
        test_case: ConversationalTestCase,
        event: VoiceEvent,
        recording: Optional[Tuple[Any, Any, int]],
    ) -> Optional[bytes]:
        return event_clip(test_case, event, self.event_speaker, recording)

    @abstractmethod
    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        raise NotImplementedError


class BaseRollUpMetric(BaseVoiceMetric):
    """One score over several single-defect metrics: the share of agent
    replies none of them flagged."""

    counted: Tuple[str, str] = AGENT_REPLIES
    reports_timing_source = False

    @abstractmethod
    def _parts(self) -> List[BaseDefectMetric]:
        raise NotImplementedError

    def _calls_a_model(self) -> bool:
        return any(part._calls_a_model() for part in self._parts())

    def _critical_events(
        self, test_case: ConversationalTestCase
    ) -> List[VoiceEvent]:
        return []

    def _prepare(self, test_case: ConversationalTestCase) -> None:
        pass

    def _part_test_case(
        self, test_case: ConversationalTestCase
    ) -> ConversationalTestCase:
        # A recording's moments can only be placed on replies that carry
        # their start time; otherwise parts judge each reply's own audio.
        if test_case.audio is not None and untimed_call_reason(test_case):
            return test_case.model_copy(update={"audio": None})
        return test_case

    def _roll_up_key(
        self,
        test_case: ConversationalTestCase,
        part: BaseDefectMetric,
        share: DefectShare,
        key: float,
        defect: Optional[VoiceEvent] = None,
    ) -> Optional[float]:
        if share.event_key != EVENT_KEYS[CALL_RECORDING]:
            return key
        stopped_at = None if defect is None else defect.get(STOPPED_AT_SECONDS)
        return agent_turn_at(
            test_case, key if stopped_at is None else stopped_at
        )

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        with shared_analysis():
            return self._roll_up(test_case)

    def _roll_up(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        part_test_case = self._part_test_case(test_case)
        critical_events = self._critical_events(part_test_case)
        if critical_events:
            return (
                0.0,
                "A critical failure occurred: "
                + ", ".join(event.type for event in critical_events)
                + ".",
                {
                    "critical_failure": True,
                    "events": events_to_dicts(critical_events),
                },
            )
        self._prepare(test_case)
        eligible: Set[float] = set()
        flagged_by: Dict[float, List[str]] = {}
        events: List[Dict] = []
        skipped: List[Dict] = []
        metrics: Dict[str, Dict] = {}
        unscored_reasons: List[str] = []
        timing_source = None
        dismissed = 0
        for part in self._parts():
            name = part.__name__
            share = part.confirmed_defect_share(part_test_case)
            dismissed += share.dismissed
            if part.evaluation_cost:
                self.evaluation_cost = (
                    self.evaluation_cost or 0.0
                ) + part.evaluation_cost
            self.evaluation_model = (
                self.evaluation_model or part.evaluation_model
            )
            for entry in share.skipped:
                if entry not in skipped:
                    skipped.append(entry)
            if not share.scored:
                metrics[name] = {
                    "score": None,
                    "reason": share.no_eligible_reason,
                }
                unscored_reasons.append(share.no_eligible_reason)
                continue
            timing_source = timing_source or share.timing_source
            # Moments that map to no key, such as ones in turns left out of
            # the test case, are dropped from the part's own counts too.
            kept_eligible: List[float] = []
            for key in share.eligible_turns:
                roll_up_key = self._roll_up_key(test_case, part, share, key)
                if roll_up_key is not None:
                    kept_eligible.append(key)
                    eligible.add(roll_up_key)
            kept_defects: List[VoiceEvent] = []
            part_flagged: Set[float] = set()
            for defect in sorted(share.defects, key=share.defect_key):
                roll_up_key = self._roll_up_key(
                    test_case, part, share, share.defect_key(defect), defect
                )
                if roll_up_key is None:
                    continue
                kept_defects.append(defect)
                events.append({**defect.to_dict(), "metric": name})
                part_flagged.add(roll_up_key)
            eligible |= part_flagged
            for roll_up_key in part_flagged:
                flagged_by.setdefault(roll_up_key, []).append(name)
            part_score, _, part_breakdown = replace(
                share, eligible_turns=kept_eligible, defects=kept_defects
            ).result()
            metrics[name] = {
                "score": part_score,
                "eligible_turns": part_breakdown["eligible_turns"],
                "turns_with_events": part_breakdown["turns_with_events"],
            }
        breakdown: Dict[str, Any] = {
            "eligible_turns": len(eligible),
            "turns_with_events": len(flagged_by),
            "events": events,
            "skipped": skipped,
            "metrics": metrics,
        }
        if self.reports_timing_source and timing_source is not None:
            breakdown["timing_source"] = timing_source
        if dismissed:
            breakdown["dismissed_events"] = dismissed
        if not eligible:
            reasons = set(unscored_reasons)
            reason = (
                reasons.pop() if len(reasons) == 1 else NO_ELIGIBLE_PARTS_REASON
            )
            return None, reason, breakdown
        if not flagged_by:
            reason = f"No {self.counted[0]} had an issue."
        else:
            counts: Dict[str, int] = {}
            for names in flagged_by.values():
                for name in names:
                    counts[name] = counts.get(name, 0) + 1
            reason = (
                f"{len(flagged_by)} of {len(eligible)} {self.counted[1]} had "
                "an issue: "
                + ", ".join(
                    f"{name} ({count})" for name, count in counts.items()
                )
                + "."
            )
        if self.reports_timing_source and timing_source == TURN_TIMESTAMPS:
            reason += TURN_TIMESTAMPS_NOTE
        reason += dismissed_note(dismissed)
        return 1.0 - len(flagged_by) / len(eligible), reason, breakdown


class BaseExchangeRollUpMetric(BaseRollUpMetric):
    """Rolls up per exchange: a caller line and the agent's reply to it."""

    counted = EXCHANGES
    use_speech_detection: bool
    _line_start_of_turn: Dict[int, int]

    def _prepare(self, test_case: ConversationalTestCase) -> None:
        self._line_start_of_turn = caller_line_start_of_turn(
            test_case, self.use_speech_detection
        )

    def _roll_up_key(
        self,
        test_case: ConversationalTestCase,
        part: BaseDefectMetric,
        share: DefectShare,
        key: float,
        defect: Optional[VoiceEvent] = None,
    ) -> Optional[int]:
        turn = super()._roll_up_key(test_case, part, share, key, defect)
        if turn is None:
            return None
        return exchange_of(test_case, int(turn), self._line_start_of_turn)
