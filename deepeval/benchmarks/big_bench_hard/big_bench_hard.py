from typing import List
from datasets import load_dataset

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask


class BigBenchHard(DeepEvalBaseBenchmark):
    def __init__(
        self, model: DeepEvalBaseLLM = None, task: BigBenchHardTask = None
    ):
        super().__init__()
        # Now call the method to load the benchmark dataset and set test cases.
        self.goldens = None

    def evaluate(self, model: DeepEvalBaseLLM, tasks: List[BigBenchHardTask]):
        for task in tasks:
            goldens = self.load_benchmark_dataset(task)
            for golden in goldens:
                self.predict(model, task, golden)

    def predict(
        self, model: DeepEvalBaseLLM, task: BigBenchHardTask, golden: Golden
    ):
        ### 1. use predefined metrics based on the task
        ### Based on the task, we MAY need a
        # - custom prompt template to confine output format
        # - predefined metrics to compare actual and expected outputs
        ### 2. use model to generate actual_output for each golden
        pass

    def load_benchmark_dataset(self, task: BigBenchHardTask) -> List[Golden]:
        # load from hugging face
        dataset = load_dataset("lukaemon/bbh", task.value)
        goldens: List[Golden] = []
        for data in dataset["test"]:
            golden = Golden(input=data["input"], expectedOutput=data["target"])
            goldens.append(golden)

        return goldens


benchmark = BigBenchHard()
benchmark.evaluate(model="", tasks=[BigBenchHardTask.CAUSAL_JUDGEMENT])
