from typing import List
from datasets import load_dataset
from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.benchmarks.truthfulqa.task import TruthfulQATask
from deepeval.scorer import Scorer
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.truthfulqa.template import TruthfulqaTemplate

class TruthfulQA(DeepEvalBaseBenchmark):
    def __init__(self, tasks: List[TruthfulQATask] = None):
        super().__init__()
        self.tasks: List[TruthfulQATask] = tasks
        self.scorer = Scorer()
    
    def evaluate(self, model: DeepEvalBaseLLM):
        for task in self.tasks:
            goldens = self.load_benchmark_dataset(task)
            total_predictions = len(goldens)
            correct_predictions = 0
            for golden in goldens:
                if self.predict(model, golden):
                    correct_predictions += 1

            print(
                f"Result for TruthfulQA (task={task.value}: {correct_predictions/total_predictions}"
            )   
    
    def predict(
        self, model: DeepEvalBaseLLM, task: TruthfulQATask, golden: Golden
    ):
        
        prompt: dict = TruthfulqaTemplate.generate_output(
            input=input, task=task
        )
        prediction = model(prompt)

        ##### 2. Define metrics IF NECESSARY to evaluate prediction #####
        ##### (Only define metrics if not found in the origianl papers) #######
        return self.scorer.exact_match_score(golden.expected_output, prediction)         

    def load_benchmark_dataset(self, task: TruthfulQATask) -> List[Golden]:
        # load from hugging face
        dataset = load_dataset("truthful_qa", task.value)
        goldens: List[Golden] = []
        for data in dataset["validation"]:
            golden = Golden(input=data["question"], expectedOutput=data["best_answer"])
            goldens.append(golden)

        return goldens
