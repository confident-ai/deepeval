from datasets import load_dataset
from typing import List, Optional, Dict, Union
from tqdm import tqdm
import pandas as pd

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.squad.task import SQuADTask
from deepeval.benchmarks.squad.template import SQuADTemplate
from deepeval.scorer import Scorer
from deepeval.benchmarks.schema import MultipleChoiceSchemaLower
from deepeval.telemetry import capture_benchmark_run
from deepeval.metrics.utils import initialize_model


class EquityMedQA(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[SQuADTask] = None,
        n_shots: int = 5,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        **kwargs,
    ):
        assert n_shots <= 5, "SQuAD only supports n_shots <= 5"
        super().__init__(**kwargs)
        self.tasks: List[SQuADTask] = (
            list(SQuADTask) if tasks is None else tasks
        )
        self.scorer = Scorer()
        self.n_shots: int = n_shots
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.evaluation_model, self.using_native_evaluation_model = (
            initialize_model(evaluation_model)
        )

    def evaluate(self, model: DeepEvalBaseLLM) -> Dict:
        with capture_benchmark_run("SQuAD", len(self.tasks)):
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []
            scores_row = []

            for task in self.tasks:
                goldens = self.load_benchmark_dataset(task)
                task_correct_predictions = 0
                task_total_predictions = len(goldens)
                overall_total_predictions += len(goldens)

                for golden in tqdm(goldens, desc=f"Processing {task.value}"):
                    prediction, score = self.predict(model, golden).values()
                    if score:
                        task_correct_predictions += 1
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

                task_accuracy = (
                    task_correct_predictions / task_total_predictions
                )
                print(
                    f"SQuAD Task Accuracy (task={task.value}): {task_accuracy}"
                )
                scores_row.append((task.value, task_accuracy))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall SQuAD Accuracy: {overall_accuracy}")

            # Create a DataFrame from task_results_data
            # Columns: 'Task', 'Input', 'Prediction', 'Score'
            self.predictions = pd.DataFrame(
                predictions_row,
                columns=[
                    "Task",
                    "Input",
                    "Prediction",
                    "Expected Output",
                    "Correct",
                ],
            )
            self.task_scores = pd.DataFrame(
                scores_row, columns=["Task", "Score"]
            )
            self.overall_score = overall_accuracy

            return overall_accuracy

    def predict(self, model: DeepEvalBaseLLM, golden: Golden) -> Dict:
        # Define prompt template
        prompt: dict = SQuADTemplate.generate_output(
            input=golden.input,
            n_shots=self.n_shots,
        )

        # Enforced model generation
        try:
            res: MultipleChoiceSchemaLower = model.generate(
                prompt=prompt, schema=MultipleChoiceSchemaLower
            )
            prediction = res.answer
        except TypeError:
            prompt += "\n\nOutput the answer, which should a text segment taken from the context."
            prediction = model.generate(prompt)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        # Define Metric
        score = self.scorer.squad_score(
            golden.input,
            prediction,
            golden.expected_output,
            self.evaluation_model,
            self.using_native_evaluation_model,
        )
        return {"prediction": prediction, "score": score}

    def load_benchmark_dataset(self, task: SQuADTask) -> List[Golden]:
        dataset = load_dataset("rajpurkar/squad", trust_remote_code=True)
        self.dataset = dataset

        # Construct test set
        test_set = dataset["validation"].filter(
            lambda data: data["title"] == task.value
        )
        goldens: List[Golden] = []
        for data in test_set:
            input = SQuADTemplate.format_question(data, include_answer=False)
            expected_output = SQuADTemplate.format_output(data)
            golden = Golden(input=input, expected_output=expected_output)
            goldens.append(golden)
        return goldens
