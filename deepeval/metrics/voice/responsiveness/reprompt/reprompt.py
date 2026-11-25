from typing import Any, Dict, List, Optional, Union

from deepeval.config.eval_mode import EvalMode
from deepeval.errors import DeepEvalError
from deepeval.metrics.utils import initialize_system_one_model
from deepeval.metrics.utils.decision import SYSTEM_ONE_YES_THRESHOLD
from deepeval.metrics.voice.utils.call_activity import (
    caller_lines,
    silence_after_line,
)
from deepeval.metrics.voice.utils.defects import (
    CALLER_LINES,
    NO_CALL_AUDIO_REASON,
    call_has_audio,
    DefectShare,
)
from deepeval.metrics.voice.utils.detectors import response_was_owed
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.models.base_model import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import NoulQuestion
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics.voice.utils.events import RANGE, VoiceEvent

# A caller speaking again sooner is pausing mid-thought, not kept waiting:
# it is the wait Response Latency's default counts as slow.
REPROMPT_MIN_WAIT_SECONDS = 4.0

REPROMPT_CALL = (
    "A phone call between an AI voice agent and a caller. Each candidate is a "
    "caller line, the silence after it with no reply from the agent, and the "
    "caller's next line."
)
REPROMPT_QUESTION = (
    "Is the caller's next line a repeat or rephrasing of their line, or a "
    "check that someone is still there?"
)
REPROMPT_YES = (
    "The caller says the same thing again, rephrases it, or checks the agent "
    'is still there, such as "hello?" or "are you there?".'
)
REPROMPT_NO = (
    "The caller carries on or finishes the same thought, says something "
    "new, or is talking to someone else."
)


def _line_text(turns: List[Turn], first: int, last: int) -> str:
    return " ".join(turn.content.strip() for turn in turns[first : last + 1])


def _agent_line_before(turns: List[Turn], index: int) -> Optional[str]:
    return next(
        (
            turn.content.strip()
            for turn in reversed(turns[:index])
            if turn.role == "assistant"
        ),
        None,
    )


class RepromptMetric(BaseDefectMetric):
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        use_speech_detection: bool = True,
        flaky: bool = False,
    ):
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.system_one_model = system_one_model
        self.use_speech_detection = use_speech_detection
        if isinstance(system_one_model, DeepEvalBaseSystemOneModel):
            self.evaluation_model = system_one_model.get_model_name()

    def _calls_a_model(self) -> bool:
        return True

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        defect_label = "to be repeated before the agent replied"
        if not call_has_audio(test_case):
            return DefectShare(
                [],
                [],
                [],
                defect_label,
                no_eligible_reason=NO_CALL_AUDIO_REASON,
                counted=CALLER_LINES,
            )
        turns = test_case.turns
        lines = caller_lines(turns, self.use_speech_detection)
        eligible_turns = []
        candidates: List[VoiceEvent] = []
        evidence: List[Dict[str, Any]] = []
        for position, (first, last) in enumerate(lines):
            if not response_was_owed(turns[last].content):
                continue
            eligible_turns.append(first)
            next_line = (
                lines[position + 1] if position + 1 < len(lines) else None
            )
            # A line never answered at all is Failed Response's to judge.
            answered_later = any(
                later.role == "assistant" for later in turns[last + 1 :]
            )
            if (
                next_line is None
                or next_line[0] != last + 1
                or not answered_later
            ):
                continue
            silence = silence_after_line(turns, last, self.use_speech_detection)
            wait = silence.get("duration_seconds")
            repeat = turns[last + 1].audio
            repeat_is_timed = (
                repeat is not None and repeat.start_time is not None
            )
            if (
                repeat_is_timed
                and wait is not None
                and wait < REPROMPT_MIN_WAIT_SECONDS
            ):
                continue
            candidates.append(
                VoiceEvent(
                    type="user_reprompted",
                    turn=first,
                    reprompt_turn=last + 1,
                    mark=RANGE,
                    description=(
                        "Caller repeated themselves"
                        if wait is None
                        else f"Caller repeated themselves after {wait:.1f}s"
                    ),
                    **silence,
                )
            )
            evidence.append(
                {
                    "agent_line_before": _agent_line_before(turns, first),
                    "caller_line": _line_text(turns, first, last),
                    "silence_seconds": (
                        None if wait is None else round(wait, 1)
                    ),
                    "caller_next_line": _line_text(turns, *next_line),
                }
            )
        confirmed = self._confirmed(evidence)
        return DefectShare(
            eligible_turns,
            [event for event, real in zip(candidates, confirmed) if real],
            [],
            defect_label,
            no_eligible_reason="No caller line needed a reply.",
            counted=CALLER_LINES,
        )

    def _confirmed(self, evidence: List[Dict[str, Any]]) -> List[bool]:
        if not evidence:
            return []
        model = self._decision_model()
        questions = {
            f"candidate_{index}": NoulQuestion(
                instructions={"candidate": item, "question": REPROMPT_QUESTION},
                true=REPROMPT_YES,
                false=REPROMPT_NO,
            )
            for index, item in enumerate(evidence)
        }
        answers, cost = model.decide(
            {"call": REPROMPT_CALL, "candidates": evidence}, questions
        )
        self.evaluation_cost = (self.evaluation_cost or 0.0) + (cost or 0.0)
        confirmed = []
        for key in questions:
            answer = answers.nouls.get(key)
            if answer is None:
                raise DeepEvalError(
                    f"{model.get_model_name()} returned no answer for "
                    f"Reprompt's {key}."
                )
            confirmed.append(answer.probability >= SYSTEM_ONE_YES_THRESHOLD)
        return confirmed

    # Built on first use, so constructing the metric needs no TypeSafe key.
    def _decision_model(self) -> DeepEvalBaseSystemOneModel:
        if not isinstance(self.system_one_model, DeepEvalBaseSystemOneModel):
            self.system_one_model = initialize_system_one_model(
                self.system_one_model, EvalMode.SYSTEM_ONE
            )
            self.evaluation_model = self.system_one_model.get_model_name()
        return self.system_one_model

    @property
    def __name__(self):
        return "Reprompt"
