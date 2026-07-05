from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Dict, List

from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    SingleTurnParams,
    ArenaTestCase,
)
from deepeval.templates.resolver import (
    MetricTemplateMethod,
    resolve_template,
)

if TYPE_CHECKING:
    from deepeval.models import DeepEvalBaseLLM


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
        return resolve_template(
            "metrics",
            template_class or self.__class__.__name__,
            method,
            multimodal=multimodal,
            strict=strict,
            **kwargs,
        )


class BaseMetric(PromptMixin):
    _required_params = List[SingleTurnParams]
    threshold: float
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
    requires_trace: bool = False
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None

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

    @abstractmethod
    def is_successful(self) -> bool:
        raise NotImplementedError

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
    threshold: float
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
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None

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

    @abstractmethod
    def is_successful(self) -> bool:
        raise NotImplementedError

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
