from typing import List
from datasets import load_dataset
from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.benchmarks.truthfulqa.task import TruthfulQATask
from deepeval.scorer import Scorer

class TruthfulQA(DeepEvalBaseBenchmark):
    def __init__(self, tasks: List[TruthfulQATask] = None):
        super().__init__()
        self.tasks: List[TruthfulQATask] = tasks
        self.scorer = Scorer()

    def load_benchmark_dataset(self, task: TruthfulQATask) -> List[Golden]:
        # load from hugging face
        dataset = load_dataset("truthful_qa", task.value)
        goldens: List[Golden] = []
        for data in dataset["validation"]:
            golden = Golden(input=data["question"], expectedOutput=data["best_answer"])
            goldens.append(golden)

        return goldens
