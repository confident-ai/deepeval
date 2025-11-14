from __future__ import annotations
from typing import List, Callable, Dict, Union, TYPE_CHECKING
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
from deepeval.optimization.execution import PromptExecutor
from deepeval.errors import DeepEvalError


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


class DeepEvalScoringAdapter(ScoringAdapter):
    """
    Scoring adapter backed by DeepEval metrics.
    Two hooks keep this adapter generic:
      - executor.run(prompts, golden) / executor.a_run(...) -> actual_output
      - build_test_case(golden, actual) -> LLM/Conversational/MLLM TestCase
    """

    def __init__(
        self,
        *,
        metrics: List[
            Union[BaseMetric, BaseConversationalMetric, BaseMultimodalMetric]
        ],
        module_selector: Callable[[Candidate], ModuleId],
        executor: PromptExecutor,
        build_test_case: Callable[
            [Union[Golden, ConversationalGolden], str],
            Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
        ],
        objective_scalar: Objective = MeanObjective(),
    ):
        self.metrics = list(metrics)
        self._module_selector = module_selector
        self._executor = executor
        self._build_test_case = build_test_case
        self._objective = objective_scalar

    def _ensure_sync_executor(self) -> None:
        if not callable(getattr(self._executor, "run", None)):
            raise DeepEvalError(
                "PromptExecutor does not implement run(). Use a_optimize() only if you provide a_run(), or supply a synchronous executor."
            )

    def _ensure_async_executor(self) -> None:
        if not callable(getattr(self._executor, "a_run", None)):
            raise DeepEvalError(
                "PromptExecutor does not implement a_run(). Use optimize() (sync) or supply an async executor implementing a_run()."
            )

    def score_on_pareto(
        self,
        candidate: Candidate,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> List[float]:
        self._ensure_sync_executor()
        scores: List[float] = []
        for golden in d_pareto:
            actual = self._executor.run(candidate.prompts, golden)
            test_case = self._build_test_case(golden, actual)
            per_metric: Dict[str, float] = {}
            for metric in self.metrics:
                score = metric.measure(test_case)
                per_metric[metric.__class__.__name__] = float(score)
            scores.append(self._objective.scalarize(per_metric))
        return scores

    def minibatch_score(
        self,
        candidate: Candidate,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float:
        self._ensure_sync_executor()
        if not minibatch:
            return 0.0
        scalarized_scores: List[float] = []
        for golden in minibatch:
            actual = self._executor.run(candidate.prompts, golden)
            test_case = self._build_test_case(golden, actual)
            per_metric: Dict[str, float] = {}
            for metric in self.metrics:
                score = metric.measure(test_case)
                per_metric[metric.__class__.__name__] = float(score)
            scalarized_scores.append(self._objective.scalarize(per_metric))
        return sum(scalarized_scores) / len(scalarized_scores)

    def minibatch_feedback(
        self,
        candidate: Candidate,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str:
        # Default μ_f: concat metric.reason across minibatch (cap length lightly)
        self._ensure_sync_executor()
        reasons: List[str] = []
        for golden in minibatch:
            actual = self._executor.run(candidate.prompts, golden)
            test_case = self._build_test_case(golden, actual)
            for metric in self.metrics:
                _ = metric.measure(test_case)
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
        return "\n---\n".join(unique[:8])

    def select_module(self, candidate: Candidate) -> ModuleId:
        return self._module_selector(candidate)

    async def a_score_on_pareto(
        self,
        candidate: Candidate,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> List[float]:
        self._ensure_async_executor()
        scores: List[float] = []
        for golden in d_pareto:
            actual = await self._executor.a_run(candidate.prompts, golden)
            test_case = self._build_test_case(golden, actual)
            per_metric: Dict[str, float] = {}
            for metric in self.metrics:
                score = await metric.a_measure(test_case)
                per_metric[metric.__class__.__name__] = float(score)
            scores.append(self._objective.scalarize(per_metric))
        return scores

    async def a_minibatch_score(
        self,
        candidate: Candidate,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float:
        self._ensure_async_executor()
        if not minibatch:
            return 0.0
        scalarized_scores: List[float] = []
        for golden in minibatch:
            actual = await self._executor.a_run(candidate.prompts, golden)
            test_case = self._build_test_case(golden, actual)
            per_metric: Dict[str, float] = {}
            for metric in self.metrics:
                score = await metric.a_measure(test_case)
                per_metric[metric.__class__.__name__] = float(score)
            scalarized_scores.append(self._objective.scalarize(per_metric))
        return sum(scalarized_scores) / len(scalarized_scores)

    async def a_minibatch_feedback(
        self,
        candidate: Candidate,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str:
        self._ensure_async_executor()
        reasons: List[str] = []
        for golden in minibatch:
            actual = await self._executor.a_run(candidate.prompts, golden)
            test_case = self._build_test_case(golden, actual)
            for metric in self.metrics:
                _ = await metric.a_measure(test_case)
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
        return "\n---\n".join(unique[:8])

    async def a_select_module(self, candidate: Candidate) -> ModuleId:
        # Keep selection policy sync.
        # this function is async here for symmetric API.
        return self._module_selector(candidate)
