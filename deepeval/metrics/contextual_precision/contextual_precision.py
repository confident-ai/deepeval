import warnings
from typing import Optional, List, Tuple, Type, Union

from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
)
from deepeval.metrics.base_metric import Verdict, YES_NO
from deepeval.metrics.utils import (
    generate_qag_verdicts,
    a_generate_qag_verdicts,
    SystemOneEvalSpec,
    SystemOneVerdictSpec,
    initialize_system_one_model,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
    RetrievedContextData,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.retrieval_context_display import id_retrieval_context
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.test_case import MLLMImage
import deepeval.metrics.contextual_precision.schema as cpschema
from deepeval.templates import make_template_class


def _contextual_precision_verdict_fields(
    retrieval_context: List[str],
    multimodal: bool,
) -> Tuple[str, Union[List[str], List[Union[str, MLLMImage]]], str]:
    document_count_str = (
        f" ({len(retrieval_context)} document"
        f"{'s' if len(retrieval_context) > 1 else ''})"
    )
    context_to_display = (
        id_retrieval_context(retrieval_context)
        if multimodal
        else retrieval_context
    )
    multimodal_note = " (which can be text or an image)" if multimodal else ""
    return document_count_str, context_to_display, multimodal_note


ContextualPrecisionTemplate = make_template_class("ContextualPrecisionMetric")


class ContextualPrecisionMetric(BaseMetric):
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
            ContextualPrecisionTemplate
        ] = ContextualPrecisionTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
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

                input = test_case.input
                expected_output = test_case.expected_output
                grouped_retrieval_context = self._group_retrieval_contexts(
                    test_case.retrieval_context
                )

                self.verdicts: List[cpschema.ContextualPrecisionVerdict] = (
                    self._generate_verdicts(
                        input,
                        expected_output,
                        grouped_retrieval_context,
                        multimodal,
                    )
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(input, multimodal)
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

            input = test_case.input
            expected_output = test_case.expected_output
            grouped_retrieval_context = self._group_retrieval_contexts(
                test_case.retrieval_context
            )

            self.verdicts: List[cpschema.ContextualPrecisionVerdict] = (
                await self._a_generate_verdicts(
                    input,
                    expected_output,
                    grouped_retrieval_context,
                    multimodal,
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(input, multimodal)
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self, input: str, multimodal: bool):
        if self.include_reason is False:
            return None

        retrieval_contexts_verdicts = [
            {"verdict": verdict.verdict, "reason": verdict.reason}
            for verdict in self.verdicts
        ]
        prompt = self._get_prompt(
            "generate_reason",
            multimodal=multimodal,
            input=input,
            verdicts=retrieval_contexts_verdicts,
            score=format(self.score, ".2f"),
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=cpschema.ContextualPrecisionScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self, input: str, multimodal: bool):
        if self.include_reason is False:
            return None

        retrieval_contexts_verdicts = [
            {"verdict": verdict.verdict, "reason": verdict.reason}
            for verdict in self.verdicts
        ]
        prompt = self._get_prompt(
            "generate_reason",
            multimodal=multimodal,
            input=input,
            verdicts=retrieval_contexts_verdicts,
            score=format(self.score, ".2f"),
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=cpschema.ContextualPrecisionScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_verdicts(
        self,
        input: str,
        expected_output: str,
        retrieval_context: List[str],
        multimodal: bool,
    ) -> List[cpschema.ContextualPrecisionVerdict]:
        doc_str, ctx_disp, mm_note = _contextual_precision_verdict_fields(
            retrieval_context, multimodal
        )
        prompt = self._get_prompt(
            "generate_verdicts",
            multimodal=multimodal,
            input=input,
            expected_output=expected_output,
            document_count_str=doc_str,
            context_to_display=ctx_disp,
            multimodal_note=mm_note,
        )

        return await a_generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=cpschema.ContextualPrecisionVerdict,
            verdicts_cls=cpschema.Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(
                input, expected_output, retrieval_context
            ),
        )

    def _generate_verdicts(
        self,
        input: str,
        expected_output: str,
        retrieval_context: List[str],
        multimodal: bool,
    ) -> List[cpschema.ContextualPrecisionVerdict]:
        doc_str, ctx_disp, mm_note = _contextual_precision_verdict_fields(
            retrieval_context, multimodal
        )
        prompt = self._get_prompt(
            "generate_verdicts",
            multimodal=multimodal,
            input=input,
            expected_output=expected_output,
            document_count_str=doc_str,
            context_to_display=ctx_disp,
            multimodal_note=mm_note,
        )

        return generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=cpschema.ContextualPrecisionVerdict,
            verdicts_cls=cpschema.Verdicts,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(
                input, expected_output, retrieval_context
            ),
        )

    def _experimental_system_one_spec(
        self, input: str, expected_output: str, retrieval_context: List[str]
    ) -> SystemOneVerdictSpec:
        return SystemOneVerdictSpec(
            instructions=self._get_prompt("_experimental_system_one_verdict"),
            items=retrieval_context,
            item_key="node",
            state={"input": input, "expected_output": expected_output},
        )

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole metric as one Jev request over
        `input`, `expected_output` and the ordered `retrieval_context`; see
        EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        return SystemOneEvalSpec(
            evaluation_params=self._required_params,
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
        )

    def _group_retrieval_contexts(
        self, retrieval_contexts: List[Union[str, RetrievedContextData]]
    ) -> List[str]:
        grouped_contexts_dict = {}
        ordered_identifiers = []

        for context in retrieval_contexts:
            if isinstance(context, RetrievedContextData):
                if context.source not in grouped_contexts_dict:
                    ordered_identifiers.append(
                        {"type": "grouped", "key": context.source}
                    )
                    grouped_contexts_dict[context.source] = []
                grouped_contexts_dict[context.source].append(context.context)
            else:
                ordered_identifiers.append(
                    {"type": "standalone", "value": context}
                )

        processed_contexts = []
        for item in ordered_identifiers:
            if item["type"] == "grouped":
                source = item["key"]
                contents = grouped_contexts_dict[source]
                combined_content = f"Source: {source}\n" + "\n---\n".join(
                    contents
                )
                processed_contexts.append(combined_content)
            else:
                processed_contexts.append(item["value"])

        return processed_contexts

    def _calculate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        # Convert verdicts to a binary list where 'yes' is 1 and others are 0
        node_verdicts = [
            1 if v.verdict == Verdict.YES else 0 for v in self.verdicts
        ]

        sum_weighted_precision_at_k = 0.0
        relevant_nodes_count = 0
        for k, is_relevant in enumerate(node_verdicts, start=1):
            # If the item is relevant, update the counter and add the weighted precision at k to the sum
            if is_relevant:
                relevant_nodes_count += 1
                precision_at_k = relevant_nodes_count / k
                sum_weighted_precision_at_k += precision_at_k * is_relevant

        if relevant_nodes_count == 0:
            return 0
        # Calculate weighted cumulative precision
        score = sum_weighted_precision_at_k / relevant_nodes_count
        return 0 if self.strict_mode and score < self.threshold else score

    @property
    def __name__(self):
        return "Contextual Precision"
