from abc import ABC, abstractmethod
from typing import Any, Optional, List

from deepeval.test_case import LLMTestCase


class DeepEvalBaseBenchmark(ABC):
    def __init__(self, *args, **kwargs):
        self._test_cases: Optional[List[LLMTestCase]] = None

    @property
    def test_cases(self) -> Optional[List[LLMTestCase]]:
        return self._test_cases

    @test_cases.setter
    def test_cases(self, test_cases: List[LLMTestCase]):
        self._test_cases = test_cases

    @abstractmethod
    def load_benchmark_dataset(self, *args, **kwargs) -> None:
        """Load the benchmark dataset and initialize test cases."""
        pass
