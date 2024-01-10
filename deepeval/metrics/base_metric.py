from abc import abstractmethod

from deepeval.test_case import LLMTestCase
from typing import Optional, Dict


class BaseMetric:
    score: float = 0
    score_metadata: Dict = None
    reason: Optional[str] = None

    @property
    def minimum_score(self) -> float:
        return self._minimum_score

    @minimum_score.setter
    def minimum_score(self, value: float):
        self._minimum_score = value

    @abstractmethod
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raise NotImplementedError

    @abstractmethod
    def is_successful(self) -> bool:
        raise NotImplementedError

    @property
    def __name__(self):
        return "Base Metric"
