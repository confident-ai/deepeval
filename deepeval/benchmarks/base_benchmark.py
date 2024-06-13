from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, List, Optional
from datasets import Dataset

from deepeval.dataset import Golden


T = TypeVar("T")


class DeepEvalBaseBenchmark(ABC, Generic[T]):
    def __init__(self, dataset: Optional[Dataset] = None):
        self.tasks: List[T] = []
        self.dataset = dataset

    @abstractmethod
    def load_benchmark_dataset(self) -> List[Golden]:
        """Load the benchmark dataset and initialize tasks."""
        raise NotImplementedError
