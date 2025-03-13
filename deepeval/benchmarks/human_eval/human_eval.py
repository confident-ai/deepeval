from typing import List, Optional, Dict, Tuple
from datasets import load_dataset
import pandas as pd

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.human_eval.task import HumanEvalTask
from deepeval.benchmarks.human_eval.template import HumanEvalTemplate
from deepeval.telemetry import capture_benchmark_run


class HumanEval(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[HumanEvalTask] = None,
        n: int = 200,
        verbose_mode: bool = False,
        **kwargs,
    ):
        from deepeval.scorer import Scorer

        super().__init__(**kwargs)
        self.tasks: List[HumanEvalTask] = (
            list(HumanEvalTask) if tasks is None else tasks
        )
        self.scorer = Scorer()
        self.temperature = 0.8
        self.n = n
        self.c = {}
        self.functions = {}
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.verbose_mode: bool = (False,)

    def evaluate(self, model: DeepEvalBaseLLM, k: int) -> Dict:
        with capture_benchmark_run("HumanEval", len(self.tasks)):
            assert self.n >= k
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []
            scores_row = []

            for task in self.tasks:
                golden: Golden = self.load_benchmark_dataset(task)
                task_correct = 0
                overall_total_predictions += 1

                # Calculate task accuracy
                prediction, score = self.predict(
                    model, task, golden, k
                ).values()
                if score:
                    task_correct = 1
                    overall_correct_predictions += 1
                predictions_row.append(
                    (
                        task.value,
                        golden.input,
                        prediction,
                        golden.expected_output,
                        score,
                    )
                )
                if self.verbose_mode:
                    self.print_verbose_logs(
                        task.value, golden.input, prediction, score
                    )
                print(
                    f"HumanEval Task Accuracy (task={task.value}): {task_correct}"
                )
                scores_row.append((task.value, task_correct))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall HumanEval Accuracy: {overall_accuracy}")

            # Create a DataFrame from task_results_data
            # Columns: 'Task', 'Input', 'Prediction', 'Score'
            self.predictions = pd.DataFrame(
                predictions_row,
                columns=[
                    "Task",
                    "Input",
                    "Prediction",
                    "Correct",
                    "Expected Output",
                    "Score",
                ],
            )
            self.task_scores = pd.DataFrame(
                scores_row, columns=["Task", "Score"]
            )
            self.overall_score = overall_accuracy

            return overall_accuracy

    def predict(
        self,
        model: DeepEvalBaseLLM,
        task: HumanEvalTask,
        golden: Golden,
        k: int,
    ) -> Dict:

        # functional correctness
        c = self.c.get(task.value, None)
        functions = self.functions.get(task.value, None)
        if c is None:
            # Define prompt template
            prompt: dict = HumanEvalTemplate.generate_output(
                input=golden.input,
                task=task,
            )
            functions = model.generate_samples(
                prompt=prompt, n=self.n, temperature=self.temperature
            )
            c = 0
            for function in functions:
                try:
                    exec(function)
                    exec(golden.expected_output)
                    c += 1
                except AssertionError as e:
                    pass
            self.c[task.value] = c
            self.functions[task.value] = functions

        # Define Metric
        score = self.scorer.pass_at_k(self.n, c, k)
        return {"prediction": functions, "score": score}

    def load_benchmark_dataset(self, task: HumanEvalTask) -> List[Golden]:
        # Cache
        if self.dataset:
            dataset = self.dataset
        else:
            dataset = load_dataset("openai_humaneval", trust_remote_code=True)
            self.dataset = dataset

        # Filter tasks
        test_set = dataset["test"].filter(
            lambda data: data["entry_point"] == task.value
        )[0]
        # Construct test set
        golden = Golden(
            input=test_set["prompt"], expected_output=test_set["test"]
        )
        return golden

    def print_verbose_logs(
        self, task_value: str, input: str, prediction: str, score: int
    ) -> str:
        steps = [
            f"Input:\n{input}",
            f"Score: {score}\nPrediction: {prediction}",
        ]
        verbose_logs = ""
        for i in range(len(steps) - 1):
            verbose_logs += steps[i]

            # don't add new line for penultimate step
            if i < len(steps) - 2:
                verbose_logs += " \n \n"

        if self.verbose_mode:
            print("*" * 50)
            print(f"Task = {task_value}")
            print("*" * 50)
            print("")
            print(verbose_logs + f"\n \n{steps[-1]}")
            print("")
            print("=" * 70)

        return verbose_logs
