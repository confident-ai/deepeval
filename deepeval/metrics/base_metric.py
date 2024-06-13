from abc import abstractmethod
from contextvars import ContextVar
from typing import Optional, Dict

from deepeval.test_case import LLMTestCase, ConversationalTestCase

class BaseMetric:

    evaluation_model: Optional[str] = None
    strict_mode: bool = False
    async_mode: bool = True
    include_reason: bool = False
    evaluation_cost: Optional[float] = None

    def __init__(self):
        self._score: ContextVar[Optional[float]] = ContextVar(f'{self.__class__.__name__}_score', default=None)
        self._score_breakdown: ContextVar[Optional[Dict]] = ContextVar(f'{self.__class__.__name__}_score_breakdown', default=None)
        self._reason: ContextVar[Optional[str]] = ContextVar(f'{self.__class__.__name__}_reason', default=None)
        self._success: ContextVar[Optional[bool]] = ContextVar(f'{self.__class__.__name__}_success', default=None)
        self._error: ContextVar[Optional[str]] = ContextVar(f'{self.__class__.__name__}_error', default=None)

    @property
    def score(self) -> Optional[float]:
        return self._score.get()
    @score.setter
    def score(self, value: Optional[float]) -> None:
        self._score.set(value)

    @property
    def score_breakdown(self) -> Optional[Dict]:
        return self._score_breakdown.get()
    @score_breakdown.setter
    def score_breakdown(self, value: Optional[Dict]) -> None:
        self._score_breakdown.set(value)

    @property
    def reason(self) -> Optional[str]:
        return self._reason.get()
    @reason.setter
    def reason(self, value: Optional[str]) -> None:
        self._reason.set(value)

    @property
    def success(self) -> Optional[bool]:
        return self._success.get()
    @success.setter
    def success(self, value: Optional[bool]) -> None:
        self._success.set(value)

    @property
    def error(self) -> Optional[str]:
        return self._error.get()
    @error.setter
    def error(self, value: Optional[str]) -> None:
        self._error.set(value)

    @property
    def error(self) -> Optional[str]:
        return self._error.get()
    @error.setter
    def error(self, value: Optional[str]) -> None:
        self._error.set(value)

    @property
    def threshold(self) -> float:
        return self._threshold
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

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


class BaseConversationalMetric:

    evaluation_model: Optional[str] = None
    # Not changeable for now
    strict_mode: bool = False
    async_mode: bool = False

    def __init__(self):
        self._score = ContextVar(f'{self.__class__.__name__}_score', default=None)
        self._score_breakdown = ContextVar(f'{self.__class__.__name__}_score_breakdown', default=None)
        self._reason = ContextVar(f'{self.__class__.__name__}_reason', default=None)
        self._error = ContextVar(f'{self.__class__.__name__}_error', default=None)

    @property
    def score(self) -> Optional[float]:
        return self._score.get()
    @score.setter
    def score(self, value: Optional[float]) -> None:
        self._score.set(value)

    @property
    def score_breakdown(self) -> Optional[Dict]:
        return self._score_breakdown.get()
    @score_breakdown.setter
    def score_breakdown(self, value: Optional[Dict]) -> None:
        self._score_breakdown.set(value)

    @property
    def reason(self) -> Optional[str]:
        return self._reason.get()
    @reason.setter
    def reason(self, value: Optional[str]) -> None:
        self._reason.set(value)

    @property
    def error(self) -> Optional[str]:
        return self._error.get()
    @error.setter
    def error(self, value: Optional[str]) -> None:
        self._error.set(value)

    @property
    def threshold(self) -> float:
        return self._threshold
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    @abstractmethod
    def measure(
        self, test_case: ConversationalTestCase, *args, **kwargs
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def is_successful(self) -> bool:
        raise NotImplementedError

    @property
    def __name__(self):
        return "Base Conversational Metric"
