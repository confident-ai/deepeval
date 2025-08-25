from deepeval.models.base_model import DeepEvalBaseLLM
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, List, Optional
from pydantic import BaseModel

from deepeval.dataset import Golden


class DeepEvalBaseBenchmarkResult(BaseModel):
    overall_accuracy: float


T = TypeVar("T")


class DeepEvalBaseBenchmark(ABC, Generic[T]):
    def __init__(self, dataset: Optional["Dataset"] = None):
        from datasets import Dataset

        self.tasks: List[T] = []
        self.dataset = dataset

    @abstractmethod
    def load_benchmark_dataset(self, *args, **kwargs) -> List[Golden]:
        """Load the benchmark dataset and initialize tasks."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, model: DeepEvalBaseLLM, *args, **kwargs
    ) -> DeepEvalBaseBenchmarkResult:
        raise NotImplementedError
