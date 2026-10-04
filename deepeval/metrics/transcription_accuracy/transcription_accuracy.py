from typing import Dict, List, Optional, Type, Union

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.base_metric import Verdict, YES_NO
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    a_generate_qag_verdicts,
    a_generate_with_schema_and_extract,
    a_run_system_one_eval,
    construct_verbose_logs,
    generate_qag_verdicts,
    generate_with_schema_and_extract,
    initialize_model,
    initialize_system_one_model,
    parse_questions,
    prepare_measure,
    run_system_one_eval,
    score_qag_verdicts,
    SystemOneEvalSpec,
    SystemOneVerdictSpec,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.metrics.transcription_accuracy.schema import (
    TranscriptionAccuracyScoreReason,
    TranscriptionAccuracyVerdict,
    Verdicts,
)
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.templates import make_template_class
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn
from deepeval.utils import get_or_create_event_loop, prettify_list


TranscriptionAccuracyTemplate = make_template_class(
    "TranscriptionAccuracyMetric"
)


def get_transcribed_exchanges(turns: List[Turn]) -> List[Dict[str, str]]:
    exchanges: List[Dict[str, str]] = []
    spoken: List[str] = []
    for turn in turns:
        if turn.role == "user":
            if turn.content:
                spoken.append(turn.content)
            continue
        if turn.provider_transcription is not None and spoken:
            exchanges.append(
                {
                    "spoken": " ".join(spoken),
                    "transcribed": turn.provider_transcription,
                    "agent_reply": turn.content,
                }
            )
        spoken = []
    return exchanges


class TranscriptionAccuracyMetric(BaseConversationalMetric):
    _required_test_case_params = [
        MultiTurnParams.CONTENT,
        MultiTurnParams.ROLE,
        MultiTurnParams.PROVIDER_TRANSCRIPTION,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[EvalModeName] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
        evaluation_template: Type[
            TranscriptionAccuracyTemplate
        ] = TranscriptionAccuracyTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.eval_mode = resolve_eval_mode(eval_mode)
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.system_one_model = initialize_system_one_model(
            system_one_model, self.eval_mode
        )
        self.evaluation_model = (
            self.model or self.system_one_model
        ).get_model_name()
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
                if run_system_one_eval(self, test_case):
                    return self.score

                self.exchanges = get_transcribed_exchanges(test_case.turns)
                self.verdicts = self._generate_verdicts()
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Exchanges:\n{prettify_list(self.exchanges)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
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
            if await a_run_system_one_eval(self, test_case):
                return self.score

            self.exchanges = get_transcribed_exchanges(test_case.turns)
            self.verdicts = await self._a_generate_verdicts()
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Exchanges:\n{prettify_list(self.exchanges)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_verdicts(
        self,
    ) -> List[TranscriptionAccuracyVerdict]:
        if len(self.exchanges) == 0:
            return []

        prompt = self._get_prompt("generate_verdicts", exchanges=self.exchanges)
        return await a_generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=TranscriptionAccuracyVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(),
        )

    def _generate_verdicts(self) -> List[TranscriptionAccuracyVerdict]:
        if len(self.exchanges) == 0:
            return []

        prompt = self._get_prompt("generate_verdicts", exchanges=self.exchanges)
        return generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=TranscriptionAccuracyVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(),
        )

    def _experimental_system_one_spec(self) -> SystemOneVerdictSpec:
        return SystemOneVerdictSpec(
            instructions=self._get_prompt("_experimental_system_one_verdict"),
            items=self.exchanges,
            item_key="exchange",
        )

    def _system_one_eval_spec(
        self, test_case: ConversationalTestCase
    ) -> Optional[SystemOneEvalSpec]:
        return SystemOneEvalSpec(
            evaluation_params=self._required_test_case_params,
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
            extra_state={
                "exchanges": get_transcribed_exchanges(test_case.turns)
            },
        )

    def _mistranscriptions(self) -> List[str]:
        return [
            verdict.reason
            for verdict in self.verdicts
            if verdict is not None
            and verdict.verdict is not None
            and verdict.verdict == Verdict.NO
        ]

    async def _a_generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None

        prompt = self._get_prompt(
            "generate_reason",
            score=self.score,
            mistranscriptions=self._mistranscriptions(),
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=TranscriptionAccuracyScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None

        prompt = self._get_prompt(
            "generate_reason",
            score=self.score,
            mistranscriptions=self._mistranscriptions(),
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=TranscriptionAccuracyScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _calculate_score(self) -> float:
        # None verdicts (failed generation / out-of-vocabulary replies) are
        # dropped by score_qag_verdicts.
        return score_qag_verdicts(self, self.verdicts, passing=(Verdict.YES,))

    @property
    def __name__(self):
        return "Transcription Accuracy"
