from abc import abstractmethod

from deepeval.test_case import LLMTestCase
from typing import Optional


class BaseMetric:
    # set an arbitrary minimum score that will get over-ridden later
    score: float = 0
    reason: Optional[str] = None

    @property
    def minimum_score(self) -> float:
        return self._minimum_score

    @minimum_score.setter
    def minimum_score(self, value: float):
        self._minimum_score = value

    # Measure function signature is subject to be different - not sure
    # how applicable this is - might need a better abstraction
    @abstractmethod
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raise NotImplementedError

    def _get_init_values(self):
        # We use this method for sending useful metadata
        init_values = {
            param: getattr(self, param)
            for param in vars(self)
            if isinstance(getattr(self, param), (str, int, float))
        }
        return init_values

    @abstractmethod
    def is_successful(self) -> bool:
        raise NotImplementedError

    @property
    def __name__(self):
        return "Base Metric"
