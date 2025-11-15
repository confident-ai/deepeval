from __future__ import annotations
import asyncio
import copy
import inspect
import json
from typing import (
    Awaitable,
    List,
    Callable,
    Dict,
    Union,
    Optional,
    TYPE_CHECKING,
)
from functools import lru_cache

from deepeval.optimization.types import (
    Candidate,
    Objective,
    MeanObjective,
    ModuleId,
    ScoringAdapter,
)
from deepeval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
    BaseMultimodalMetric,
)
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    MLLMTestCase,
)
from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalBaseLLM
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


class DeepEvalScoringAdapter(ScoringAdapter):
    """Scoring adapter backed by DeepEval metrics with a built-in generation step.
    Users provide either a `model` (DeepEvalBaseLLM) or a `generate(prompt, golden)` callable.
    """

    def __init__(
        self,
        *,
        metrics: List[
            Union[BaseMetric, BaseConversationalMetric, BaseMultimodalMetric]
        ],
        build_test_case: Optional[
            Callable[
                [Union[Golden, ConversationalGolden], str],
                Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
            ]
        ] = None,
        model: Optional[DeepEvalBaseLLM] = None,
        model_schema: Optional[PydanticBaseModel] = None,
        generate: Optional[
            Callable[[Prompt, "Golden"], Union[str, Awaitable[str]]]
        ] = None,
        objective_scalar: Objective = MeanObjective(),
    ):
        self.metrics = list(metrics)
        self._model = model
        self._model_schema = model_schema
        self._generate = generate
        self._build_test_case = build_test_case or self._default_build_test_case
        self._objective = objective_scalar

        # async
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._throttle: float = 0.0

        if self._model is None and self._generate is None:
            raise DeepEvalError(
                "DeepEvalScoringAdapter requires either `model=` or `generate=`."
            )

    #######################################
    # prompt assembly & result unwrapping #
    #######################################
    def _compile_prompt_text(self, prompt: Prompt, golden: "Golden") -> str:
        user_input = getattr(golden, "input", None) or ""
        # Keep it simple and predictable; metrics do judging separately.
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
        # Generic LLM case; conversational/multimodal users can pass their own builder
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
        candidate: Candidate,
        golden: Union[Golden, ConversationalGolden],
    ) -> float:
        # Clone metrics to avoid shared-state races (e.g., .reason)
        metrics = [copy.copy(metric) for metric in self.metrics]
        actual = await self.a_generate(candidate.prompts, golden)
        test_case = self._build_test_case(golden, actual)
        per_metric: Dict[str, float] = {}
        for metric in metrics:
            score = await _a_measure_no_indicator(metric, test_case)
            per_metric[metric.__class__.__name__] = float(score)
        return self._objective.scalarize(per_metric)

    def _score_one(
        self,
        candidate: Candidate,
        golden: Union[Golden, ConversationalGolden],
    ) -> float:
        metrics = [copy.copy(m) for m in self.metrics]
        actual = self.generate(candidate.prompts, golden)
        test_case = self._build_test_case(golden, actual)
        per_metric: Dict[str, float] = {}
        for metric in metrics:
            score = _measure_no_indicator(metric, test_case)
            per_metric[metric.__class__.__name__] = float(score)
        return self._objective.scalarize(per_metric)

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
        prompt = prompts_by_module.get("__module__") or next(
            iter(prompts_by_module.values())
        )
        compiled = self._compile_prompt_text(prompt, golden)
        if self._generate:
            out = self._generate(prompt, golden)
            return out if isinstance(out, str) else str(out)
        res = self._model.generate(compiled, schema=self._model_schema)
        return self._unwrap_text(res)

    async def a_generate(
        self, prompts_by_module: Dict[ModuleId, Prompt], golden: "Golden"
    ) -> str:
        prompt = prompts_by_module.get("__module__") or next(
            iter(prompts_by_module.values())
        )
        compiled = self._compile_prompt_text(prompt, golden)
        if self._generate:
            maybe = self._generate(prompt, golden)
            res = await maybe if inspect.isawaitable(maybe) else maybe
            return self._unwrap_text(res)
        if hasattr(self._model, "a_generate"):
            res = await self._model.a_generate(
                compiled, schema=self._model_schema
            )
            return self._unwrap_text(res)
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: self._model.generate(compiled, schema=self._model_schema),
        )
        return self._unwrap_text(res)

    def score_on_pareto(
        self,
        candidate: Candidate,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> List[float]:
        return [self._score_one(candidate, golden) for golden in d_pareto]

    def minibatch_score(
        self,
        candidate: Candidate,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float:
        if not minibatch:
            return 0.0

        scores = [self._score_one(candidate, golden) for golden in minibatch]
        return sum(scores) / len(scores)

    def minibatch_feedback(
        self,
        candidate: Candidate,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str:
        # default metric feedback (μ_f): concat metric.reason across minibatch and cap length lightly
        reasons: List[str] = []
        for golden in minibatch:
            actual = self.generate(candidate.prompts, golden)
            test_case = self._build_test_case(golden, actual)
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
        return "\n---\n".join(unique[:8])  # TODO: Make this configurable

    async def a_score_on_pareto(
        self,
        candidate: Candidate,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> List[float]:
        tasks = [
            self._bounded(self._a_score_one(candidate, golden))
            for golden in d_pareto
        ]
        return await asyncio.gather(*tasks)

    async def a_minibatch_score(
        self,
        candidate: Candidate,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float:
        tasks = [
            self._bounded(self._a_score_one(candidate, golden))
            for golden in minibatch
        ]
        scores = await asyncio.gather(*tasks)
        return sum(scores) / len(scores) if scores else 0.0

    async def a_minibatch_feedback(
        self,
        candidate: Candidate,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str:
        async def reasons_one(golden) -> List[str]:
            # Clone per task to avoid data races on .reason
            metrics = [copy.copy(metric) for metric in self.metrics]
            # metrics = self.metrics
            actual = await self.a_generate(candidate.prompts, golden)
            test_case = self._build_test_case(golden, actual)
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

    def select_module(self, candidate: Candidate) -> ModuleId:
        return "__module__"

    async def a_select_module(self, candidate: Candidate) -> ModuleId:
        return "__module__"
