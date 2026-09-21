from __future__ import annotations

import copy
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from deepeval.templates.resolver import (
    MetricTemplateMethod,
    resolve_template,
)
from deepeval.templates.template_class import filter_template_kwargs
from deepeval.test_case import (
    ArenaTestCase,
    ConversationalTestCase,
    LLMTestCase,
    SingleTurnParams,
)

if TYPE_CHECKING:
    from deepeval.models import DeepEvalBaseLLM


_RUN_STATE = {
    "score",
    "reason",
    "success",
    "error",
    "verdicts",
    "skipped",
    "evaluation_cost",
    "input_tokens",
    "output_tokens",
    "verbose_logs",
}


def _clone_metric(metric):
    copied = copy.copy(metric)

    for key, value in vars(copied).items():
        if key not in _RUN_STATE and isinstance(value, (list, dict, set)):
            setattr(copied, key, copy.copy(value))

    for key in _RUN_STATE:
        if hasattr(copied, key):
            setattr(copied, key, None)

    return copied


class PromptMixin:
    """Renders a metric prompt template. `template_class` overrides the default
    `self.__class__.__name__` when borrowing another class's templates."""

    def _get_prompt(
        self,
        method: MetricTemplateMethod,
        *,
        template_class: Optional[str] = None,
        multimodal: bool = False,
        strict: bool = True,
        **kwargs,
    ) -> str:
        context = {**kwargs, "multimodal": multimodal, "strict": strict}

        # An explicit `template_class` borrows another class's templates, so an
        # `evaluation_template` set for this metric must not hijack it.
        if template_class is None:
            render = getattr(
                getattr(self, "evaluation_template", None), method, None
            )
            if render is not None:
                return render(**filter_template_kwargs(render, context))

        return resolve_template(
            "metrics",
            template_class or self.__class__.__name__,
            method,
            **context,
        )


class BaseMetric(PromptMixin):
    _required_params = List[SingleTurnParams]
    threshold: Optional[float] = None
    score: Optional[float] = None
    score_breakdown: Dict = None
    reason: Optional[str] = None
    success: Optional[bool] = None
    evaluation_model: Optional[str] = None
    strict_mode: bool = False
    async_mode: bool = True
    verbose_mode: bool = True
    include_reason: bool = False
    error: Optional[str] = None
    evaluation_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    verbose_logs: Optional[str] = None
    skipped = False
    flaky: bool = False
    requires_trace: bool = False
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None

    def clone(self) -> "BaseMetric":
        """Return a per-test-case copy without duplicating the model client."""
        return _clone_metric(self)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from deepeval.tracing.internal import observe_methods

        observe_methods(cls)

    @abstractmethod
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raise NotImplementedError

    @abstractmethod
    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet. Please set 'async_mode' to 'False'."
        )

    def is_successful(self) -> Optional[bool]:
        if self.threshold is None:
            self.success = None
        elif self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Base Metric"

    def _accrue_cost(self, cost: Optional[float]) -> None:
        effective = getattr(cost, "value", cost)
        if self.evaluation_cost is not None and effective is not None:
            self.evaluation_cost += effective
        else:
            self.evaluation_cost = None

    def _accrue_tokens(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens


class BaseConversationalMetric(PromptMixin):
    threshold: Optional[float] = None
    score: Optional[float] = None
    score_breakdown: Dict = None
    reason: Optional[str] = None
    success: Optional[bool] = None
    evaluation_model: Optional[str] = None
    strict_mode: bool = False
    async_mode: bool = True
    verbose_mode: bool = True
    include_reason: bool = False
    error: Optional[str] = None
    evaluation_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    verbose_logs: Optional[str] = None
    skipped = False
    flaky: bool = False
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None

    def clone(self) -> "BaseConversationalMetric":
        """Return a per-test-case copy without duplicating the model client."""
        return _clone_metric(self)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from deepeval.tracing.internal import observe_methods

        observe_methods(cls)

    @abstractmethod
    def measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    async def a_measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet. Please set 'async_mode' to 'False'."
        )

    def is_successful(self) -> Optional[bool]:
        if self.threshold is None:
            self.success = None
        elif self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Base Conversational Metric"

    def _accrue_cost(self, cost: Optional[float]) -> None:
        effective = getattr(cost, "value", cost)
        if self.evaluation_cost is not None and effective is not None:
            self.evaluation_cost += effective
        else:
            self.evaluation_cost = None

    def _accrue_tokens(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens


class BaseArenaMetric(PromptMixin):
    reason: Optional[str] = None
    evaluation_model: Optional[str] = None
    async_mode: bool = True
    verbose_mode: bool = True
    include_reason: bool = False
    error: Optional[str] = None
    evaluation_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    verbose_logs: Optional[str] = None
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from deepeval.tracing.internal import observe_methods

        observe_methods(cls)

    @abstractmethod
    def measure(self, test_case: ArenaTestCase, *args, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    async def a_measure(self, test_case: ArenaTestCase, *args, **kwargs) -> str:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet. Please set 'async_mode' to 'False'."
        )

    @abstractmethod
    def is_successful(self) -> bool:
        raise NotImplementedError

    @property
    def __name__(self):
        return "Base Arena Metric"

    def _accrue_cost(self, cost: Optional[float]) -> None:
        effective = getattr(cost, "value", cost)
        if self.evaluation_cost is not None and effective is not None:
            self.evaluation_cost += effective
        else:
            self.evaluation_cost = None

    def _accrue_tokens(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens
        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens
