from typing import List, Optional, Union, Type
import asyncio

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
)
from deepeval.metrics.base_metric import Verdict, YES_NO_BORDERLINE
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
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    SystemOneEvalSpec,
    SystemOneVerdictSpec,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.faithfulness.schema import (
    FaithfulnessVerdict,
    Verdicts,
    FaithfulnessScoreReason,
    Truths,
    Claims,
)
from deepeval.templates import make_template_class


def _faithfulness_truths_limit_phrase(extraction_limit: Optional[int]) -> str:
    if extraction_limit is None:
        return " FACTUAL, undisputed truths"
    if extraction_limit == 1:
        return " the single most important FACTUAL, undisputed truth"
    return f" the {extraction_limit} most important FACTUAL, undisputed truths per document"


def _faithfulness_truths_multimodal_instruction(multimodal: bool) -> str:
    if multimodal:
        return " The excerpt may contain both text and images."
    return ""


def _faithfulness_claims_multimodal_instruction(multimodal: bool) -> str:
    if multimodal:
        return (
            " The excerpt may contain both text and images, so extract claims from "
            "all provided content."
        )
    return ""


FaithfulnessTemplate = make_template_class("FaithfulnessMetric")


class FaithfulnessMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
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
        truths_extraction_limit: Optional[int] = None,
        penalize_ambiguous_claims: bool = False,
        flaky: bool = False,
        evaluation_template: Type[FaithfulnessTemplate] = FaithfulnessTemplate,
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
        self.penalize_ambiguous_claims = penalize_ambiguous_claims

        self.truths_extraction_limit = truths_extraction_limit
        if self.truths_extraction_limit is not None:
            self.truths_extraction_limit = max(self.truths_extraction_limit, 0)
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
            multimodal,
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

                retrieval_context = test_case.retrieval_context
                actual_output = test_case.actual_output

                self.truths = self._generate_truths(
                    retrieval_context, multimodal
                )
                self.claims = self._generate_claims(actual_output, multimodal)
                self.verdicts = self._generate_verdicts(multimodal)
                self.score = self._calculate_score()
                self.reason = self._generate_reason(multimodal)
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Truths (limit={self.truths_extraction_limit}):\n{prettify_list(self.truths)}",
                        f"Claims:\n{prettify_list(self.claims)}",
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
            multimodal,
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

            retrieval_context = test_case.retrieval_context
            actual_output = test_case.actual_output

            self.truths, self.claims = await asyncio.gather(
                self._a_generate_truths(retrieval_context, multimodal),
                self._a_generate_claims(actual_output, multimodal),
            )
            self.verdicts = await self._a_generate_verdicts(multimodal)
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(multimodal)
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Truths (limit={self.truths_extraction_limit}):\n{prettify_list(self.truths)}",
                    f"Claims:\n{prettify_list(self.claims)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self, multimodal: bool) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict == Verdict.NO:
                contradictions.append(verdict.reason)
            if (
                verdict.verdict == Verdict.BORDERLINE
                and self.penalize_ambiguous_claims
            ):
                contradictions.append(f"(Ambiguous) {verdict.reason}")

        prompt = self._get_prompt(
            "generate_reason",
            multimodal=multimodal,
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=FaithfulnessScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self, multimodal: bool) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict == Verdict.NO:
                contradictions.append(verdict.reason)
            if (
                verdict.verdict == Verdict.BORDERLINE
                and self.penalize_ambiguous_claims
            ):
                contradictions.append(f"(Ambiguous) {verdict.reason}")

        prompt = self._get_prompt(
            "generate_reason",
            multimodal=multimodal,
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=FaithfulnessScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_verdicts(
        self, multimodal: bool
    ) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        prompt = self._get_prompt(
            "generate_verdicts",
            multimodal=multimodal,
            claims=self.claims,
            retrieval_context="\n\n".join(self.truths),
        )

        return await a_generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=FaithfulnessVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO_BORDERLINE,
            system_one=self._experimental_system_one_spec(),
        )

    def _generate_verdicts(self, multimodal: bool) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        prompt = self._get_prompt(
            "generate_verdicts",
            multimodal=multimodal,
            claims=self.claims,
            retrieval_context="\n\n".join(self.truths),
        )

        return generate_qag_verdicts(
            metric=self,
            prompt=prompt,
            verdict_cls=FaithfulnessVerdict,
            verdicts_cls=Verdicts,
            allowed=YES_NO_BORDERLINE,
            system_one=self._experimental_system_one_spec(),
        )

    def _experimental_system_one_spec(self) -> SystemOneVerdictSpec:
        return SystemOneVerdictSpec(
            instructions=self._get_prompt("_experimental_system_one_verdict"),
            items=self.claims,
            item_key="claim",
            state={"retrieval_context": self.truths},
        )

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole metric as one Jev request over
        `actual_output` and `retrieval_context`; see EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        return SystemOneEvalSpec(
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
        )

    async def _a_generate_truths(
        self, retrieval_context: str, multimodal: bool
    ) -> List[str]:
        prompt = self._get_prompt(
            "generate_truths",
            multimodal=multimodal,
            retrieval_context="\n\n".join(retrieval_context),
            limit=_faithfulness_truths_limit_phrase(
                self.truths_extraction_limit
            ),
            multimodal_instruction=_faithfulness_truths_multimodal_instruction(
                multimodal
            ),
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Truths,
            extract_schema=lambda s: s.truths,
            extract_json=lambda data: data["truths"],
        )

    def _generate_truths(
        self, retrieval_context: str, multimodal: bool
    ) -> List[str]:
        prompt = self._get_prompt(
            "generate_truths",
            multimodal=multimodal,
            retrieval_context="\n\n".join(retrieval_context),
            limit=_faithfulness_truths_limit_phrase(
                self.truths_extraction_limit
            ),
            multimodal_instruction=_faithfulness_truths_multimodal_instruction(
                multimodal
            ),
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Truths,
            extract_schema=lambda s: s.truths,
            extract_json=lambda data: data["truths"],
        )

    async def _a_generate_claims(
        self, actual_output: str, multimodal: bool
    ) -> List[str]:
        prompt = self._get_prompt(
            "generate_claims",
            multimodal=multimodal,
            actual_output=actual_output,
            multimodal_instruction=_faithfulness_claims_multimodal_instruction(
                multimodal
            ),
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Claims,
            extract_schema=lambda s: s.claims,
            extract_json=lambda data: data["claims"],
        )

    def _generate_claims(
        self, actual_output: str, multimodal: bool
    ) -> List[str]:
        prompt = self._get_prompt(
            "generate_claims",
            multimodal=multimodal,
            actual_output=actual_output,
            multimodal_instruction=_faithfulness_claims_multimodal_instruction(
                multimodal
            ),
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Claims,
            extract_schema=lambda s: s.claims,
            extract_json=lambda data: data["claims"],
        )

    def _calculate_score(self) -> float:
        passing = (
            (Verdict.YES,)
            if self.penalize_ambiguous_claims
            else (Verdict.YES, Verdict.BORDERLINE)
        )
        return score_qag_verdicts(self, self.verdicts, passing=passing)

    @property
    def __name__(self):
        return "Faithfulness"
