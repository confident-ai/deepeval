from typing import List, Optional, Dict, Union
from tqdm import tqdm

from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BiasMetric
from deepeval.benchmarks.base_benchmark import (
    DeepEvalBaseBenchmark,
    DeepEvalBaseBenchmarkResult,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.equity_med_qa.task import EquityMedQATask
from deepeval.benchmarks.equity_med_qa.template import EquityMedQATemplate
from deepeval.telemetry import capture_benchmark_run
from deepeval.metrics.utils import initialize_model


class EquityMedQA(DeepEvalBaseBenchmark):
    def __init__(
        self,
        tasks: List[EquityMedQATask] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        **kwargs,
    ):
        from deepeval.scorer import Scorer
        import pandas as pd

        super().__init__(**kwargs)
        self.tasks: List[EquityMedQATask] = (
            list(EquityMedQATask) if tasks is None else tasks
        )
        self.scorer = Scorer()
        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None
        self.evaluation_model, self.using_native_evaluation_model = (
            initialize_model(model)
        )

    def evaluate(
        self, model: DeepEvalBaseLLM, *args, **kwargs
    ) -> DeepEvalBaseBenchmarkResult:
        import pandas as pd

        with capture_benchmark_run("EquityMedQA", len(self.tasks)):
            overall_correct_predictions = 0
            overall_total_predictions = 0
            predictions_row = []
            scores_row = []

            for task in self.tasks:
                goldens = self.load_benchmark_dataset(task)
                task_correct_predictions = 0
                task_total_predictions = len(goldens)
                overall_total_predictions += len(goldens)

                for golden in tqdm(
                    goldens[:10], desc=f"Processing {task.value}"
                ):
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
                    f"EquityMedQA Task Accuracy (task={task.value}): {task_accuracy}"
                )
                scores_row.append((task.value, task_accuracy))

            # Calculate overall accuracy
            overall_accuracy = (
                overall_correct_predictions / overall_total_predictions
            )
            print(f"Overall EquityMedQA Accuracy: {overall_accuracy}")

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

            return DeepEvalBaseBenchmarkResult(
                overall_accuracy=overall_accuracy
            )

    def predict(self, model: DeepEvalBaseLLM, golden: Golden) -> Dict:
        prediction = model.generate(golden.input)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        # Define Metric
        metric = BiasMetric(
            model=self.evaluation_model,
            strict_mode=True,
        )
        score = metric.measure(
            LLMTestCase(input=golden.input, actual_output=prediction),
            _show_indicator=False,
            _log_metric_to_confident=False,
        )
        flipped_score = (
            1 - metric.score if metric.score in [0, 1] else metric.score
        )
        return {"prediction": prediction, "score": int(flipped_score)}

    def load_benchmark_dataset(self, task: EquityMedQATask) -> List[Golden]:
        from datasets import load_dataset

        # Load full dataset
        dataset_mapping = {
            EquityMedQATask.EHAI: EquityMedQATask.EHAI.value + "_dataset",
            EquityMedQATask.FBRT_LLM: EquityMedQATask.FBRT_LLM.value
            + "_dataset",
            EquityMedQATask.FBRT_LLM_661_SAMPLED: EquityMedQATask.FBRT_LLM_661_SAMPLED.value
            + "_dataset",
            EquityMedQATask.FBRT_MANUAL: EquityMedQATask.FBRT_MANUAL.value
            + "_dataset",
            EquityMedQATask.MIXED_MMQA_OMAQ: EquityMedQATask.MIXED_MMQA_OMAQ.value
            + "_dataset",
            EquityMedQATask.MULTIMEDQA: EquityMedQATask.MULTIMEDQA.value
            + "_dataset",
            EquityMedQATask.OMAQ: EquityMedQATask.OMAQ.value + "_dataset",
            EquityMedQATask.OMIYE_ET_AL: EquityMedQATask.OMIYE_ET_AL.value
            + "_dataset",
            EquityMedQATask.TRINDS: EquityMedQATask.TRINDS.value + "_dataset",
        }
        dataset_attr = dataset_mapping.get(task)
        if dataset_attr:
            if not hasattr(self, dataset_attr):
                dataset = load_dataset("katielink/EquityMedQA", task.value)
                setattr(self, dataset_attr, dataset)
            else:
                dataset = getattr(self, dataset_attr)

        # Construct test set
        goldens: List[Golden] = []
        for data in dataset["train"]:
            input = EquityMedQATemplate.format_question(data)
            golden = Golden(input=input)
            goldens.append(golden)
        return goldens
