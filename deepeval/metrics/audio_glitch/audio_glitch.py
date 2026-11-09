import json
from typing import Dict, List, Optional, Type, Union

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    a_generate_with_schema_and_extract,
    construct_verbose_logs,
    generate_with_schema_and_extract,
    initialize_model,
    prepare_measure,
)
from deepeval.metrics.audio_glitch._detectors import (
    GlitchReport,
    detect_audio_glitches,
)
from deepeval.metrics.audio_glitch.schema import AudioGlitchReason
from deepeval.models import DeepEvalBaseLLM
from deepeval.templates import make_template_class
from deepeval.test_case import ConversationalTestCase, MultiTurnParams
from deepeval.utils import get_or_create_event_loop, prettify_list


AudioGlitchTemplate = make_template_class("AudioGlitchMetric")

DEFAULT_CLEAN_REASON = (
    "No audio glitches were detected: every turn with audio was free of "
    "held tones, dead stream, stream cuts and clipping."
)


class AudioGlitchMetric(BaseConversationalMetric):
    _required_test_case_params: List[MultiTurnParams] = [
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
        MultiTurnParams.AUDIO,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
        evaluation_template: Type[AudioGlitchTemplate] = AudioGlitchTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.evaluation_template = evaluation_template

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
                self._detect(test_case)
                self.reason = self._generate_reason(test_case)
                self._finish()
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
            self._detect(test_case)
            self.reason = await self._a_generate_reason(test_case)
            self._finish()
            return self.score

    def _detect(self, test_case: ConversationalTestCase) -> None:
        self.report: GlitchReport = detect_audio_glitches(test_case)
        self.score = self._calculate_score()
        self.score_breakdown = {
            "turns_with_audio": len(self.report.turns_with_audio),
            "turns_with_events": len(
                {event["turn"] for event in self.report.events}
            ),
            "events": self.report.events,
            "baselines": self.report.baselines,
            "timeline_available": self.report.timeline_available,
        }

    def _finish(self) -> None:
        self.success = self.is_successful()
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                "Speaker baselines:\n"
                + json.dumps(self.report.baselines, indent=2),
                f"Events:\n{prettify_list(self.report.events)}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    def _calculate_score(self) -> float:
        turns_with_events = {event["turn"] for event in self.report.events}
        score = 1 - len(turns_with_events) / len(self.report.turns_with_audio)
        return 0.0 if self.strict_mode and score < 1 else score

    def _reason_events(self, test_case: ConversationalTestCase) -> List[Dict]:
        turns = test_case.turns
        rows = []
        for event in self.report.events:
            previous = turns[event["turn"] - 1] if event["turn"] > 0 else None
            rows.append(
                {
                    **event,
                    "turn_text": turns[event["turn"]].content,
                    "previous_turn_role": previous.role if previous else None,
                    "previous_turn_text": (
                        previous.content if previous else None
                    ),
                }
            )
        return rows

    def _reason_prompt(self, test_case: ConversationalTestCase) -> str:
        return self._get_prompt(
            "generate_reason",
            score=self.score,
            events=self._reason_events(test_case),
        )

    async def _a_generate_reason(
        self, test_case: ConversationalTestCase
    ) -> Optional[str]:
        if self.include_reason is False:
            return None
        if not self.report.events:
            return DEFAULT_CLEAN_REASON

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=self._reason_prompt(test_case),
            schema_cls=AudioGlitchReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(
        self, test_case: ConversationalTestCase
    ) -> Optional[str]:
        if self.include_reason is False:
            return None
        if not self.report.events:
            return DEFAULT_CLEAN_REASON

        return generate_with_schema_and_extract(
            metric=self,
            prompt=self._reason_prompt(test_case),
            schema_cls=AudioGlitchReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    @property
    def __name__(self):
        return "Audio Glitch"
