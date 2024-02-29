from abc import ABC, abstractmethod
from typing import Any, Optional, List

from deepeval.test_case import LLMTestCase
from deepeval.dataset import Golden


class DeepEvalBaseBenchmark(ABC):
    def __init__(self, *args, **kwargs):
        self._test_cases: Optional[List[LLMTestCase]] = None
        self._goldens: Optional[List[Golden]] = None

    @property
    def goldens(self) -> Optional[List[Golden]]:
        return self._goldens

    @goldens.setter
    def goldens(self, goldens: List[Golden]):
        self._goldens = goldens

    @property
    def test_cases(self) -> Optional[List[LLMTestCase]]:
        return self._test_cases

    @test_cases.setter
    def test_cases(self, test_cases: List[LLMTestCase]):
        self._test_cases = test_cases

    @abstractmethod
    def load_benchmark_dataset(self, *args, **kwargs) -> List[Golden]:
        """Load the benchmark dataset and initialize goldens."""
        raise NotImplementedError
