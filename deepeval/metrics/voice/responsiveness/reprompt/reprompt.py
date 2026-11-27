import re
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
    call_has_audio,
    no_turn_audio_reason,
    DefectShare,
)
from deepeval.metrics.voice.utils.detectors import response_was_owed
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.models.base_model import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import NoulQuestion
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics.voice.utils.events import RANGE, VoiceEvent

# A reprompt is the caller saying it again after waiting in silence; said
# sooner, they are still talking.
REPROMPT_MIN_WAIT_SECONDS = 2.5
# A bare greeting only checks that someone is there after a long silence:
# the wait Response Latency's default counts as slow.
GREETING_MIN_WAIT_SECONDS = 4.0
GREETINGS = {"hi", "hello", "hey", "hi there", "hello there", "hey there"}
# The agent's own words on the caller's line are an echo, not the caller.
ECHOED_LINE_MIN_WORDS = 3
ECHOED_LINE_MIN_WORD_SHARE = 0.8
REPROMPT_CALL = (
    "A phone call between an AI voice agent and a caller. Each candidate is a "
    "caller line, the silence after it with no reply from the agent, and the "
    "caller's next line."
)
REPROMPT_QUESTION = (
    "Does the caller's next line only say their line again, or check that "
    "someone is still there, without adding anything new?"
)
REPROMPT_YES = (
    "The next line adds no new information: it repeats the line, rewords the "
    "same request or answer, or asks whether anyone is there, such as "
    '"hello?" or "are you there?".'
)
REPROMPT_NO = (
    "The next line adds something new, even when it starts by repeating the "
    "line: more detail, the rest of an unfinished sentence, a different "
    "answer, thanks or a goodbye. Or it is unrelated, or talks to someone "
    "else."
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


def _words(text: str) -> List[str]:
    return re.sub(r"[^\w']+", " ", text.lower()).split()


def _is_greeting(text: str) -> bool:
    return " ".join(_words(text)) in GREETINGS


def _is_echo_of(line: str, agent_line: Optional[str]) -> bool:
    words, agent_words = _words(line), set(_words(agent_line or ""))
    return (
        len(words) >= ECHOED_LINE_MIN_WORDS
        and sum(word in agent_words for word in words) / len(words)
        >= ECHOED_LINE_MIN_WORD_SHARE
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
                no_eligible_reason=no_turn_audio_reason(test_case),
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
            # The wait is only known when the repeat itself is timed.
            next_turn = turns[next_line[0]]
            if (
                wait is None
                or next_turn.audio is None
                or next_turn.audio.start_time is None
                or wait < REPROMPT_MIN_WAIT_SECONDS
            ):
                continue
            repeat = _line_text(turns, *next_line)
            if _is_greeting(repeat) and wait < GREETING_MIN_WAIT_SECONDS:
                continue
            agent_line = _agent_line_before(turns, next_line[0])
            if _is_echo_of(repeat, agent_line) or _is_echo_of(
                _line_text(turns, first, last), agent_line
            ):
                continue
            candidates.append(
                VoiceEvent(
                    type="user_reprompted",
                    turn=first,
                    reprompt_turn=last + 1,
                    mark=RANGE,
                    description=f"Caller repeated themselves after {wait:.1f}s",
                    **silence,
                )
            )
            evidence.append(
                {
                    "agent_line_before": _agent_line_before(turns, first),
                    "caller_line": _line_text(turns, first, last),
                    "silence_seconds": round(wait, 1),
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
