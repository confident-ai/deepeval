import textwrap
from typing import List, Optional, Union

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    a_generate_with_schema_and_extract,
    construct_verbose_logs,
    generate_with_schema_and_extract,
    initialize_model,
    prepare_measure,
)
from deepeval.metrics.voice_style.schema import VoiceStyleVerdict
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import Audio, ConversationalTestCase, MultiTurnParams
from deepeval.utils import get_or_create_event_loop

# Clips sent to the judge: enough to hear the voice across the call without
# sending the whole recording.
MAX_CLIPS = 6


class VoiceStyleMetric(BaseConversationalMetric):
    """Whether the agent's voice followed a style instruction, such as
    "speak slowly and warmly".

    A model that can hear audio listens to the agent's replies, so the score
    is about how the agent sounds, not what the transcript says.
    """

    _required_test_case_params = [MultiTurnParams.ROLE, MultiTurnParams.AUDIO]

    def __init__(
        self,
        style: str,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if not style or not style.strip():
            raise ValueError(
                "`style` must describe how the agent should sound."
            )
        self.style = style
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        if self.model.supports_audio() is False:
            raise ValueError(
                f"{self.evaluation_model} cannot take audio. VoiceStyleMetric "
                "needs a model that can hear the agent's replies."
            )
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case)
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
            else:
                clips = self._clips(test_case)
                verdict = generate_with_schema_and_extract(
                    metric=self,
                    prompt=self._prompt(len(clips)),
                    schema_cls=VoiceStyleVerdict,
                    extract_schema=lambda verdict: verdict,
                    extract_json=lambda data: VoiceStyleVerdict(**data),
                    audio=clips,
                )
                self._finish(verdict, len(clips))
            return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case)
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            clips = self._clips(test_case)
            verdict = await a_generate_with_schema_and_extract(
                metric=self,
                prompt=self._prompt(len(clips)),
                schema_cls=VoiceStyleVerdict,
                extract_schema=lambda verdict: verdict,
                extract_json=lambda data: VoiceStyleVerdict(**data),
                audio=clips,
            )
            self._finish(verdict, len(clips))
            return self.score

    @staticmethod
    def _clips(test_case: ConversationalTestCase) -> List[Audio]:
        spoken = [
            turn.audio
            for turn in test_case.turns
            if turn.role == "assistant" and turn.audio is not None
        ]
        if not spoken:
            raise ValueError(
                "VoiceStyleMetric needs audio on at least one assistant turn."
            )
        if len(spoken) <= MAX_CLIPS:
            return spoken
        step = (len(spoken) - 1) / (MAX_CLIPS - 1)
        return [spoken[round(index * step)] for index in range(MAX_CLIPS)]

    def _prompt(self, clips: int) -> str:
        return textwrap.dedent(
            f"""You are judging how a voice agent SOUNDS, not what it says.

            The agent was told to speak in this style:
            <<<
            {self.style}
            >>>

            You are given {clips} audio clip(s) of the agent's replies from one phone call, in order.
            Listen to the pace, tone, warmth, energy and clarity of the voice. Ignore the words' meaning
            and any mistakes in the task; judge only whether the delivery follows the style above.

            Score from 0 to 10:
            - 10: every clip clearly follows the style.
            - 5: the style is followed in part, or only in some clips.
            - 0: the delivery ignores or contradicts the style.

            IMPORTANT: The output must be formatted as a JSON object with two keys:
            - `score`: an integer from 0 to 10.
            - `reason`: one or two sentences on what you heard that led to the score.

            Example JSON Output:
            {{
                "score": 4,
                "reason": "The voice was warm, but it spoke quickly in the last three replies."
            }}

            JSON Output:
            """
        )

    def _finish(self, verdict: VoiceStyleVerdict, clips: int) -> None:
        score = max(0, min(10, verdict.score)) / 10
        self.score = (
            0.0 if self.strict_mode and score < self.threshold else score
        )
        self.reason = verdict.reason if self.include_reason else None
        self.success = self.is_successful()
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Style: {self.style}",
                f"Clips heard: {clips}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    @property
    def __name__(self):
        return "Voice Style"
