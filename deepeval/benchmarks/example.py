from typing import List

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM


class ExampleBenchmark(DeepEvalBaseBenchmark):
    def __init__(self, model: DeepEvalBaseLLM):
        super().__init__()
        # Now call the method to load the benchmark dataset and set test cases.
        self.goldens = self.load_benchmark_dataset()
        self.model = model

    def load_benchmark_dataset(self) -> List[Golden]:
        # load from hugging face
        pass
