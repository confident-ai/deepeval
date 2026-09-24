from typing import Any, Dict, Optional, List, Union, Type

from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
)
from deepeval.metrics.base_metric import Verdict, YES_NO
from deepeval.metrics.utils import (
    generate_qag_verdicts,
    a_generate_qag_verdicts,
    score_qag_verdicts,
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    initialize_system_one_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
    SystemOneEvalSpec,
    SystemOneVerdictSpec,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    split_sentences,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.retrieval_context_display import id_retrieval_context
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.contextual_recall.schema import (
    ContextualRecallVerdict,
    Verdicts,
    ContextualRecallScoreReason,
    VerdictWithExpectedOutput,
)
from deepeval.templates import make_template_class


def _contextual_recall_verdict_kwargs(
    retrieval_context: List[Any],
    multimodal: bool,
) -> Dict[str, object]:
    content_type = "sentence and image" if multimodal else "sentence"
    content_type_plural = "sentences and images" if multimodal else "sentences"
    content_or = "sentence or image" if multimodal else "sentence"
    context_to_display = (
        id_retrieval_context(retrieval_context)
        if multimodal
        else retrieval_context
    )
    node_instruction = ""
    if multimodal:
        node_instruction = (
            " A node is either a string or image, but not both (so do not group "
            "images and texts in the same nodes)."
        )
    return {
        "content_type": content_type,
        "content_type_plural": content_type_plural,
        "content_or": content_or,
        "context_to_display": context_to_display,
        "node_instruction": node_instruction,
    }


ContextualRecallTemplate = make_template_class("ContextualRecallMetric")


class ContextualRecallMetric(BaseMetric):

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
        SingleTurnParams.EXPECTED_OUTPUT,
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
            ContextualRecallTemplate
        ] = ContextualRecallTemplate,
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
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        multimodal = test_case.multimodal

        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
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

                expected_output = test_case.expected_output
                retrieval_context = test_case.retrieval_context

                self.verdicts: List[VerdictWithExpectedOutput] = (
                    self._generate_verdicts(
                        expected_output, retrieval_context, multimodal
                    )
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(expected_output, multimodal)
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:

        multimodal = test_case.multimodal

        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if await a_run_system_one_eval(self, test_case):
                return self.score

            expected_output = test_case.expected_output
            retrieval_context = test_case.retrieval_context

            self.verdicts: List[VerdictWithExpectedOutput] = (
                await self._a_generate_verdicts(
                    expected_output, retrieval_context, multimodal
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(
                expected_output, multimodal
            )
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self, expected_output: str, multimodal: bool):
        if self.include_reason is False:
            return None

        supportive_reasons = []
        unsupportive_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict == Verdict.YES:
                supportive_reasons.append(verdict.reason)
            else:
                unsupportive_reasons.append(verdict.reason)

        prompt = self._get_prompt(
            "generate_reason",
            expected_output=expected_output,
            supportive_reasons=supportive_reasons,
            unsupportive_reasons=unsupportive_reasons,
            score=format(self.score, ".2f"),
            multimodal=multimodal,
            content_type="sentence or image" if multimodal else "sentence",
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ContextualRecallScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self, expected_output: str, multimodal: bool):
        if self.include_reason is False:
            return None

        supportive_reasons = []
        unsupportive_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict == Verdict.YES:
                supportive_reasons.append(verdict.reason)
            else:
                unsupportive_reasons.append(verdict.reason)

        prompt = self._get_prompt(
            "generate_reason",
            expected_output=expected_output,
            supportive_reasons=supportive_reasons,
            unsupportive_reasons=unsupportive_reasons,
            score=format(self.score, ".2f"),
            multimodal=multimodal,
            content_type="sentence or image" if multimodal else "sentence",
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ContextualRecallScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _calculate_score(self):
        return score_qag_verdicts(
            self,
            self.verdicts,
            passing=(Verdict.YES,),
            empty_score=0,
        )

    async def _a_generate_verdicts(
        self,
        expected_output: str,
        retrieval_context: List[str],
        multimodal: bool,
    ) -> List[VerdictWithExpectedOutput]:
        prompt = self._get_prompt(
            "generate_verdicts",
            expected_output=expected_output,
            multimodal=multimodal,
            **_contextual_recall_verdict_kwargs(retrieval_context, multimodal),
        )
        verdicts = await a_generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=ContextualRecallVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(
                expected_output, retrieval_context, multimodal
            ),
        )
        final_verdicts = []
        for verdict in verdicts:
            new_verdict = VerdictWithExpectedOutput(
                verdict=verdict.verdict,
                reason=verdict.reason,
                expected_output=expected_output,
            )
            final_verdicts.append(new_verdict)
        return final_verdicts

    def _generate_verdicts(
        self,
        expected_output: str,
        retrieval_context: List[str],
        multimodal: bool,
    ) -> List[VerdictWithExpectedOutput]:
        prompt = self._get_prompt(
            "generate_verdicts",
            expected_output=expected_output,
            multimodal=multimodal,
            **_contextual_recall_verdict_kwargs(retrieval_context, multimodal),
        )
        verdicts = generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=ContextualRecallVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(
                expected_output, retrieval_context, multimodal
            ),
        )
        final_verdicts = []
        for verdict in verdicts:
            new_verdict = VerdictWithExpectedOutput(
                verdict=verdict.verdict,
                reason=verdict.reason,
                expected_output=expected_output,
            )
            final_verdicts.append(new_verdict)
        return final_verdicts

    def _experimental_system_one_spec(
        self,
        expected_output: str,
        retrieval_context: List[str],
        multimodal: bool,
    ) -> Optional[SystemOneVerdictSpec]:
        if multimodal:
            return None
        return SystemOneVerdictSpec(
            instructions=self._get_prompt("_experimental_system_one_verdict"),
            items=split_sentences(expected_output),
            item_key="sentence",
            state={"retrieval_context": retrieval_context},
        )

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole metric as one Jev request over
        `expected_output` and `retrieval_context`; see EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        return SystemOneEvalSpec(
            evaluation_params=[
                SingleTurnParams.EXPECTED_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
        )

    @property
    def __name__(self):
        return "Contextual Recall"
