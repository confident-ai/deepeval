from __future__ import annotations
import asyncio
import copy
import inspect
import json
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from functools import lru_cache

from deepeval.errors import DeepEvalError
from deepeval.optimization.types import (
    PromptConfiguration,
    Objective,
    MeanObjective,
    ModuleId,
)
from deepeval.optimization.utils import (
    validate_callback,
    validate_metrics,
    invoke_model_callback,
    a_invoke_model_callback,
    build_model_callback_kwargs,
)
from deepeval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
)
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    MLLMTestCase,
)
from deepeval.prompt.prompt import Prompt
from pydantic import BaseModel as PydanticBaseModel


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


@lru_cache(maxsize=None)
def _has_kwarg(func: Callable, keyword: str) -> bool:
    """Return True if func accepts keyword or has **kwargs."""
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        return False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters


def _measure_no_indicator(metric, test_case):
    """Call metric.measure(test_case) with _show_indicator=False if supported."""
    measure = getattr(metric, "measure")
    if _has_kwarg(measure, "_show_indicator"):
        return measure(test_case, _show_indicator=False)
    return measure(test_case)


async def _a_measure_no_indicator(metric, test_case):
    """
    Prefer metric.a_measure with fall back to metric.measure in a thread.
    Always disable indicators when supported. This is to prevent interference
    with the gepa indicator.
    """
    a_measure = getattr(metric, "a_measure", None)

    if a_measure is not None:
        call = (
            a_measure(test_case, _show_indicator=False)
            if _has_kwarg(a_measure, "_show_indicator")
            else a_measure(test_case)
        )
        # Be resilient if impl returns a plain value
        return await call if inspect.isawaitable(call) else call

    # No async impl: run sync measure in a thread
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: _measure_no_indicator(metric, test_case)
    )


class DeepEvalScoringAdapter:
    """Scoring adapter backed by DeepEval metrics with a built-in generation step."""

    DEFAULT_MODULE_ID: ModuleId = "__module__"

    def __init__(
        self,
        *,
        build_test_case: Optional[
            Callable[
                [Union[Golden, ConversationalGolden], str],
                Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
            ]
        ] = None,
        objective_scalar: Objective = MeanObjective(),
    ):
        self.model_callback: Optional[
            Callable[
                ...,
                Union[
                    str,
                    Dict,
                    Tuple[Union[str, Dict], float],
                ],
            ]
        ] = None
        self.metrics: Union[
            List[BaseMetric], List[BaseConversationalMetric]
        ] = []

        self.build_test_case = build_test_case or self._default_build_test_case
        self.objective_scalar = objective_scalar

        # async
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._throttle: float = 0.0

    def set_model_callback(
        self,
        model_callback: Callable[
            ...,
            Union[
                str,
                Dict,
                Tuple[Union[str, Dict], float],
            ],
        ],
    ):
        self.model_callback = validate_callback(
            component="DeepEvalScoringAdapter",
            model_callback=model_callback,
        )

    def set_metrics(
        self,
        metrics: Union[List[BaseMetric], List[BaseConversationalMetric]],
    ):
        self.metrics = validate_metrics(
            component="DeepEvalScoringAdapter", metrics=metrics
        )

    #######################################
    # prompt assembly & result unwrapping #
    #######################################
    def _compile_prompt_text(self, prompt: Prompt, golden: "Golden") -> str:
        user_input = getattr(golden, "input", None) or ""
        return f"{prompt.text_template}\n\n{user_input}".strip()

    def _unwrap_text(
        self, result: Union[str, Dict, PydanticBaseModel, tuple]
    ) -> str:
        # DeepEval LLMs return (output, cost), unwrap if so.
        if isinstance(result, tuple) and result:
            result = result[0]
        if isinstance(result, PydanticBaseModel):
            return result.model_dump_json()
        if isinstance(result, dict):
            return json.dumps(result)
        return str(result)

    #####################
    # Test case helpers #
    #####################
    def _default_build_test_case(
        self, golden: "Golden", actual: str
    ) -> LLMTestCase:
        # Generic LLM case
        return LLMTestCase(
            input=getattr(golden, "input", None),
            expected_output=getattr(golden, "expected_output", None),
            actual_output=actual,
            context=getattr(golden, "context", None),
            retrieval_context=getattr(golden, "retrieval_context", None),
        )

    ###################
    # scoring helpers #
    ###################

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
        metrics = [copy.copy(metric) for metric in self.metrics]
        actual = await self.a_generate(prompt_configuration.prompts, golden)
        test_case = self.build_test_case(golden, actual)
        per_metric: Dict[str, float] = {}
        for metric in metrics:
            score = await _a_measure_no_indicator(metric, test_case)
            per_metric[metric.__class__.__name__] = float(score)
        return self.objective_scalar.scalarize(per_metric)

    def _score_one(
        self,
        prompt_configuration: PromptConfiguration,
        golden: Union[Golden, ConversationalGolden],
    ) -> float:
        metrics = [copy.copy(m) for m in self.metrics]
        actual = self.generate(prompt_configuration.prompts, golden)
        test_case = self.build_test_case(golden, actual)
        per_metric: Dict[str, float] = {}
        for metric in metrics:
            score = _measure_no_indicator(metric, test_case)
            per_metric[metric.__class__.__name__] = float(score)
        return self.objective_scalar.scalarize(per_metric)

    #################
    # Configuration #
    #################

    def configure_async(
        self, *, max_concurrent: int = 20, throttle_seconds: float = 0.0
    ):
        # The runner will call this once, but it is safe to recreate between runs
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._throttle = float(throttle_seconds)

    ########################
    # generation & scoring #
    ########################

    def generate(
        self, prompts_by_module: Dict[ModuleId, Prompt], golden: "Golden"
    ) -> str:

        if not prompts_by_module:
            raise DeepEvalError(
                "DeepEvalScoringAdapter.generate(...) received an empty "
                "`prompts_by_module`; at least one Prompt is required."
            )

        validate_callback(
            component="DeepEvalScoringAdapter",
            model_callback=self.model_callback,
        )
        validate_metrics(
            component="DeepEvalScoringAdapter", metrics=self.metrics
        )

        module_id = self._select_module_id_from_prompts(prompts_by_module)
        prompt = prompts_by_module.get(module_id) or next(
            iter(prompts_by_module.values())
        )
        prompt_text = self._compile_prompt_text(prompt, golden)

        candidate_kwargs = build_model_callback_kwargs(
            prompt=prompt,
            prompt_text=prompt_text,
            golden=golden,
        )
        result = invoke_model_callback(
            hook="score_generate",
            model_callback=self.model_callback,
            candidate_kwargs=candidate_kwargs,
        )

        return self._unwrap_text(result)

    async def a_generate(
        self, prompts_by_module: Dict[ModuleId, Prompt], golden: "Golden"
    ) -> str:

        if not prompts_by_module:
            raise DeepEvalError(
                "DeepEvalScoringAdapter.a_generate(...) received an empty "
                "`prompts_by_module`; at least one Prompt is required."
            )

        validate_callback(
            component="DeepEvalScoringAdapter",
            model_callback=self.model_callback,
        )
        validate_metrics(
            component="DeepEvalScoringAdapter", metrics=self.metrics
        )

        module_id = self._select_module_id_from_prompts(prompts_by_module)
        prompt = prompts_by_module.get(module_id) or next(
            iter(prompts_by_module.values())
        )
        prompt_text = self._compile_prompt_text(prompt, golden)

        candidate_kwargs = build_model_callback_kwargs(
            prompt=prompt,
            prompt_text=prompt_text,
            golden=golden,
        )
        result = await a_invoke_model_callback(
            hook="score_generate",
            model_callback=self.model_callback,
            candidate_kwargs=candidate_kwargs,
        )
        return self._unwrap_text(result)

    def score_on_pareto(
        self,
        prompt_configuration: PromptConfiguration,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> List[float]:
        return [
            self._score_one(prompt_configuration, golden) for golden in d_pareto
        ]

    def minibatch_score(
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

    def minibatch_feedback(
        self,
        prompt_configuration: PromptConfiguration,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str:
        # default metric feedback (μ_f): concat metric.reason across minibatch and cap length
        reasons: List[str] = []
        for golden in minibatch:
            actual = self.generate(prompt_configuration.prompts, golden)
            test_case = self.build_test_case(golden, actual)
            for metric in [copy.copy(m) for m in self.metrics]:
                _ = _measure_no_indicator(metric, test_case)
                if getattr(metric, "reason", None):
                    reasons.append(str(metric.reason))
        if not reasons:
            return ""
        unique: List[str] = []
        seen = set()
        for reason in reasons:
            if reason not in seen:
                unique.append(reason)
                seen.add(reason)
        return "\n---\n".join(
            unique[:8]
        )  # TODO: Make how much feedback configurable

    async def a_score_on_pareto(
        self,
        prompt_configuration: PromptConfiguration,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> List[float]:
        tasks = [
            self._bounded(self._a_score_one(prompt_configuration, golden))
            for golden in d_pareto
        ]
        return await asyncio.gather(*tasks)

    async def a_minibatch_score(
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

    async def a_minibatch_feedback(
        self,
        prompt_configuration: PromptConfiguration,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str:
        async def reasons_one(golden) -> List[str]:
            # Clone per task to avoid shared state
            metrics = [copy.copy(metric) for metric in self.metrics]
            # metrics = self.metrics
            actual = await self.a_generate(prompt_configuration.prompts, golden)
            test_case = self.build_test_case(golden, actual)
            out: List[str] = []
            for metric in metrics:
                _ = await _a_measure_no_indicator(metric, test_case)
                if getattr(metric, "reason", None):
                    out.append(str(metric.reason))
            return out

        tasks = [self._bounded(reasons_one(golden)) for golden in minibatch]
        nested = await asyncio.gather(*tasks)
        reasons: List[str] = [reason for sub in nested for reason in sub]
        if not reasons:
            return ""
        unique: List[str] = []
        seen = set()
        for reason in reasons:
            if reason not in seen:
                unique.append(reason)
                seen.add(reason)
        return "\n---\n".join(unique[:8])

    def _select_module_id_from_prompts(
        self, prompts_by_module: Dict[ModuleId, Prompt]
    ) -> ModuleId:
        """
        Default module selection strategy:

        - Prefer the synthetic '__module__' key when present
        - Otherwise fall back to the first key in prompts_by_module.

        Assumes `prompts_by_module` is non-empty; callers should validate that.
        """
        if self.DEFAULT_MODULE_ID in prompts_by_module:
            return self.DEFAULT_MODULE_ID

        # At this point we expect at least one key.
        try:
            return next(iter(prompts_by_module.keys()))
        except StopIteration:
            raise DeepEvalError(
                "DeepEvalScoringAdapter._select_module_id_from_prompts(...) "
                "received an empty `prompts_by_module`. At least one Prompt is required."
            )

    def select_module(
        self, prompt_configuration: PromptConfiguration
    ) -> ModuleId:
        return self._select_module_id_from_prompts(prompt_configuration.prompts)

    async def a_select_module(
        self, prompt_configuration: PromptConfiguration
    ) -> ModuleId:
        return self.select_module(prompt_configuration)
