from abc import abstractmethod
from typing import Optional, Dict, List

from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    LLMTestCaseParams,
    ArenaTestCase,
)
from deepeval.models import DeepEvalBaseLLM


class BaseMetric:
    _required_params = List[LLMTestCaseParams]
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
    verbose_logs: Optional[str] = None
    skipped = False
    requires_trace: bool = False
    model = Optional[DeepEvalBaseLLM]
    using_native_model = Optional[bool]

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

    def _accrue_cost(self, cost: float) -> None:
        if self.evaluation_cost is not None and cost is not None:
            self.evaluation_cost += cost
        else:
            self.evaluation_cost = None


class BaseConversationalMetric:
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
    verbose_logs: Optional[str] = None
    skipped = False
    model: Optional[DeepEvalBaseLLM] = None
    using_native_model: Optional[bool] = None

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

    def _accrue_cost(self, cost: float) -> None:
        if self.evaluation_cost is not None and cost is not None:
            self.evaluation_cost += cost
        else:
            self.evaluation_cost = None


class BaseArenaMetric:
    reason: Optional[str] = None
    evaluation_model: Optional[str] = None
    async_mode: bool = True
    verbose_mode: bool = True
    include_reason: bool = False
    error: Optional[str] = None
    evaluation_cost: Optional[float] = None
    verbose_logs: Optional[str] = None
    model = Optional[DeepEvalBaseLLM]
    using_native_model = Optional[bool]

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

    def _accrue_cost(self, cost: float) -> None:
        if self.evaluation_cost is not None and cost is not None:
            self.evaluation_cost += cost
        else:
            self.evaluation_cost = None
