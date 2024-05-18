from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, List

from deepeval.dataset import Golden


T = TypeVar("T")


class DeepEvalBaseBenchmark(ABC, Generic[T]):
    def __init__(self):
        self.tasks: List[T] = []

    @abstractmethod
    def load_benchmark_dataset(self) -> List[Golden]:
        """Load the benchmark dataset and initialize tasks."""
        raise NotImplementedError
