from __future__ import annotations
import asyncio
import copy
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from deepeval.dataset.golden import Golden, ConversationalGolden
from deepeval.dataset.utils import (
    convert_goldens_to_test_cases,
    convert_convo_goldens_to_convo_test_cases,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.errors import DeepEvalError
from deepeval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
)
from deepeval.metrics.utils import copy_metrics
from deepeval.prompt.api import PromptType
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    Turn,
)
from deepeval.prompt.prompt import Prompt
from deepeval.metrics.utils import a_generate_with_schema_and_extract, generate_with_schema_and_extract, initialize_model

from deepeval.optimizer.types import (
    ModelCallback,
    PromptConfiguration,
    ModuleId,
)
from deepeval.optimizer.scorer.base import BaseScorer
from deepeval.optimizer.utils import (
    validate_callback,
    validate_metrics,
    invoke_model_callback,
    a_invoke_model_callback,
)
from deepeval.optimizer.scorer.utils import (
    _measure_no_indicator,
    _a_measure_no_indicator,
)
from .template import ScorerTemplate
from .schema import ScorerDiagnosisResult, ScorerDiagnosisSchema


class Scorer(BaseScorer):
    """
    Scores prompts by running model_callback, building test cases,
    running metrics, and aggregating scores.
    """

    DEFAULT_MODULE_ID: ModuleId = "__module__"

    def __init__(
        self,
        model_callback: ModelCallback,
        metrics: Union[List[BaseMetric], List[BaseConversationalMetric]],
        max_concurrent: int,
        throttle_seconds: float,
        optimizer_model: DeepEvalBaseLLM,
    ):
        self.model_callback = validate_callback(
            component="Scorer",
            model_callback=model_callback,
        )
        self.metrics = validate_metrics(component="Scorer", metrics=metrics)
        self.model, self.using_native_model = initialize_model(optimizer_model)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._throttle = float(throttle_seconds)

    ########################
    # generation & scoring #
    ########################

    def generate(
        self,
        prompts_by_module: Dict[ModuleId, Prompt],
        golden: Union[Golden, ConversationalGolden],
    ) -> str:
        module_id = self._select_module_id_from_prompts(prompts_by_module)
        prompt = prompts_by_module.get(module_id) or next(
            iter(prompts_by_module.values())
        )

        return invoke_model_callback(
            model_callback=self.model_callback,
            prompt=prompt,
            golden=golden,
        )

    async def a_generate(
        self,
        prompts_by_module: Dict[ModuleId, Prompt],
        golden: Union[Golden, ConversationalGolden],
    ) -> str:
        module_id = self._select_module_id_from_prompts(prompts_by_module)
        prompt = prompts_by_module.get(module_id) or next(
            iter(prompts_by_module.values())
        )

        return await a_invoke_model_callback(
            model_callback=self.model_callback,
            prompt=prompt,
            golden=golden,
        )

    def score_pareto(
        self,
        prompt_configuration: PromptConfiguration,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> List[float]:
        return [
            self._score_one(prompt_configuration, golden) for golden in d_pareto
        ]

    def score_minibatch(
        self,
        prompt_configuration: PromptConfiguration,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float:
        if not minibatch:
            return 0.0

        scores = [
            self._score_one(prompt_configuration, golden)
            for golden in minibatch
        ]
        return sum(scores) / len(scores)

    def get_minibatch_feedback(
        self,
        prompt_configuration: PromptConfiguration,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> ScorerDiagnosisResult:
        results: List[str] = []
        for golden in minibatch:
            actual = self.generate(prompt_configuration.prompts, golden)
            test_case = self._golden_to_test_case(golden, actual)
            
            metrics = copy_metrics(self.metrics)
            for metric in metrics:
                _measure_no_indicator(metric=metric, test_case=test_case)
            
            evaluation_results_block = self._build_evaluation_results_block(golden, actual, metrics)
            if evaluation_results_block:
                results.append(evaluation_results_block)
        
        if not results:
            return ScorerDiagnosisResult(
                failures=[],
                successes=[],
                analysis="",
                results=[],
            )
        
        evaluation_results = "\n\n---\n\n".join(results)

        prompt = prompt_configuration.prompts[module]
        original_prompt = prompt.text_template if prompt.type == PromptType.TEXT else prompt.messages_template

        diagnosis_prompt = ScorerTemplate.generate_diagnosis(
            original_prompt=original_prompt,
            evaluation_results=evaluation_results,
        )

        diagnosis = generate_with_schema_and_extract(
            metric=self,
            prompt=diagnosis_prompt,
            schema_cls=ScorerDiagnosisSchema,
            extract_schema=lambda s: s,
            extract_json=lambda data: data,
        )
        return ScorerDiagnosisResult(
            failures=diagnosis.failures,
            successes=diagnosis.successes,
            analysis=diagnosis.analysis,
            results=results,
        )
        

    async def a_score_pareto(
        self,
        prompt_configuration: PromptConfiguration,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> List[float]:
        tasks = [
            self._bounded(self._a_score_one(prompt_configuration, golden))
            for golden in d_pareto
        ]
        return await asyncio.gather(*tasks)

    async def a_score_minibatch(
        self,
        prompt_configuration: PromptConfiguration,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float:
        tasks = [
            self._bounded(self._a_score_one(prompt_configuration, golden))
            for golden in minibatch
        ]
        scores = await asyncio.gather(*tasks)
        return sum(scores) / len(scores) if scores else 0.0

    async def a_get_minibatch_feedback(
        self,
        prompt_configuration: PromptConfiguration,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> ScorerDiagnosisResult:
        async def process_one_trace(golden) -> Optional[str]:
            actual = await self.a_generate(prompt_configuration.prompts, golden)
            test_case = self._golden_to_test_case(golden, actual)
            
            metrics = copy_metrics(self.metrics)
            for metric in metrics:
                await _a_measure_no_indicator(metric=metric, test_case=test_case)
            
            return self._build_evaluation_results_block(golden, actual, metrics)

        tasks = [self._bounded(process_one_trace(golden)) for golden in minibatch]
        raw_results = await asyncio.gather(*tasks)
        
        results = [r for r in raw_results if r]
        
        if not results:
            return ScorerDiagnosisResult(
                failures=[],
                successes=[],
                analysis="",
                results=[],
            )
        
        evaluation_results = "\n\n---\n\n".join(results)

        prompt = prompt_configuration.prompts[module]
        original_prompt = prompt.text_template if prompt.type == PromptType.TEXT else prompt.messages_template

        diagnosis_prompt = ScorerTemplate.generate_diagnosis(
            original_prompt=original_prompt,
            evaluation_results=evaluation_results,
        )

        diagnosis = await a_generate_with_schema_and_extract(
            metric=self,
            prompt=diagnosis_prompt,
            schema_cls=ScorerDiagnosisSchema,
            extract_schema=lambda s: s,
            extract_json=lambda data: data,
        )
        return ScorerDiagnosisResult(
            failures=diagnosis.failures,
            successes=diagnosis.successes,
            analysis=diagnosis.analysis,
            results=results,
        )

    ###################
    # scoring helpers #
    ###################

    def _golden_to_test_case(
        self,
        golden: Union[Golden, ConversationalGolden],
        actual: str,
    ) -> Union[LLMTestCase, ConversationalTestCase]:
        """Convert a golden + actual output into a test case for metrics."""
        if isinstance(golden, Golden):
            golden.actual_output = actual
            return convert_goldens_to_test_cases([golden])[0]

        if isinstance(golden, ConversationalGolden):
            # Build turns with actual output as assistant response
            turns: List[Turn] = list(golden.turns or [])
            if turns and turns[-1].role == "assistant":
                turns[-1] = Turn(role="assistant", content=actual)
            elif turns:
                turns.append(Turn(role="assistant", content=actual))
            else:
                turns = [
                    Turn(role="assistant", content=actual),
                ]

            golden.turns = turns
            return convert_convo_goldens_to_convo_test_cases([golden])[0]

    async def _bounded(self, coro):
        if self._semaphore is None:
            return await coro
        async with self._semaphore:
            res = await coro
        if self._throttle:
            await asyncio.sleep(self._throttle)
        return res

    async def _a_score_one(
        self,
        prompt_configuration: PromptConfiguration,
        golden: Union[Golden, ConversationalGolden],
    ) -> float:
        # Clone metrics to avoid shared-state
        metrics = copy_metrics(self.metrics)
        actual = await self.a_generate(prompt_configuration.prompts, golden)
        test_case = self._golden_to_test_case(golden, actual)

        per_metric: Dict[str, float] = {}
        for metric in metrics:
            score = await _a_measure_no_indicator(metric, test_case)
            per_metric[metric.__class__.__name__] = float(score)
        score = sum(per_metric.values()) / len(per_metric) if per_metric else 0.0
        return score

    def _score_one(
        self,
        prompt_configuration: PromptConfiguration,
        golden: Union[Golden, ConversationalGolden],
    ) -> float:
        metrics = copy_metrics(self.metrics)
        actual = self.generate(prompt_configuration.prompts, golden)
        test_case = self._golden_to_test_case(golden, actual)

        per_metric: Dict[str, float] = {}
        for metric in metrics:
            score = _measure_no_indicator(metric, test_case)
            per_metric[metric.__class__.__name__] = float(score)
        score = sum(per_metric.values()) / len(per_metric) if per_metric else 0.0
        return score

    def _select_module_id_from_prompts(
        self, prompts_by_module: Dict[ModuleId, Prompt]
    ) -> ModuleId:
        if self.DEFAULT_MODULE_ID in prompts_by_module:
            return self.DEFAULT_MODULE_ID

        # At this point we expect at least one key.
        try:
            return next(iter(prompts_by_module.keys()))
        except StopIteration:
            raise DeepEvalError(
                "Scorer._select_module_id_from_prompts(...) "
                "received an empty `prompts_by_module`. At least one Prompt is required."
            )

    def _build_evaluation_results_block(
        self, 
        golden: Union[Golden, ConversationalGolden], 
        actual: str, 
        metrics: List[BaseMetric]
    ) -> str:
        if isinstance(golden, Golden):
            input_str = golden.input
            expected_str = golden.expected_output or "None provided"
        else:
            input_str = "\n".join([t.content for t in golden.turns if t.role == "user"])
            expected_str = golden.expected_outcome or "None provided"
        
        reasons = []
        for metric in metrics:
            score = metric.score
            reason = metric.reason
            reasons.append(f"- {metric.__class__.__name__} (Score: {score}): {reason}")
            
        return (
            f"[Input]: {input_str}\n"
            f"[Expected]: {expected_str}\n"
            f"[Actual Model Output]: {actual}\n"
            f"[Evaluation Reasons]:\n" + "\n".join(reasons)
        )
