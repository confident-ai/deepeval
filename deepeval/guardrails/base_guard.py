from abc import ABC, abstractmethod
from typing import Optional, Dict
from deepeval.guardrails.types import GuardType


class BaseDecorativeGuard(ABC):
    score: Optional[float] = None
    score_breakdown: Dict = None
    reason: Optional[str] = None
    evaluation_model: Optional[str] = None
    error: Optional[str] = None
    latency: Optional[float] = None
    guard_type: GuardType

    @property
    def __name__(self):
        return "Base Decorative Guard"


class BaseGuard(BaseDecorativeGuard):
    @abstractmethod
    async def a_guard_input(self, input: str, *args, **kwargs) -> float:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet."
        )

    @abstractmethod
    async def a_guard_output(
        self, input: str, output: str, *args, **kwargs
    ) -> float:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet."
        )

    @property
    def __name__(self):
        return "Base Guard"
