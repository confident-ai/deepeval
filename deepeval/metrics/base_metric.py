from abc import abstractmethod

from deepeval.test_case import LLMTestCase
from typing import Optional, Dict


class BaseMetric:
    score: float = 0
    score_breakdown: Dict = None
    reason: Optional[str] = None
    evaluation_model: Optional[str] = None

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
    def is_successful(self) -> bool:
        raise NotImplementedError

    @property
    def __name__(self):
        return "Base Metric"
