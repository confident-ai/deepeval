from typing import Union, List, Dict
from .utils import get_column_order, generate_test_cases
from .rich_manager import RichManager

from deepeval.metrics import BaseMetric
from deepeval.evaluate import execute_test_cases
from deepeval.dataset import EvaluationDataset

try:
    from transformers import (
        TrainerCallback,
        ProgressCallback,
        Trainer,
        TrainingArguments,
        TrainerState,
        TrainerControl,
    )

    class DeepEvalHuggingFaceCallback(TrainerCallback):
        """
        Custom callback for deep evaluation during model training.

        Args:
            metrics (List[BaseMetric]): List of evaluation metrics.
            evaluation_dataset (EvaluationDataset): Dataset for evaluation.
            tokenizer_args (Dict): Arguments for the tokenizer.
            aggregation_method (str): Method for aggregating metric scores.
            trainer (Trainer): Model trainer.
        """

        def __init__(
            self,
            trainer: Trainer,
            evaluation_dataset: EvaluationDataset = None,
            metrics: List[BaseMetric] = None,
            tokenizer_args: Dict = None,
            aggregation_method: str = "avg",
            show_table: bool = False,
        ) -> None:
            super().__init__()

            self.show_table = show_table
            self.metrics = metrics
            self.evaluation_dataset = evaluation_dataset
            self.tokenizer_args = tokenizer_args
            self.aggregation_method = aggregation_method
            self.trainer = trainer

            self.task_descriptions = {
                "generating": "[blue][STATUS] [white]Generating output from model (might take up few minutes)",
                "training": "[blue][STATUS] [white]Training in Progress",
                "evaluate": "[blue][STATUS] [white]Evaluating test-cases (might take up few minutes)",
                "training_end": "[blue][STATUS] [white]Training Ended",
            }

            self.train_bar_started = False
            self.epoch_counter = 0
            self.deepeval_metric_history = []

            total_train_epochs = self.trainer.args.num_train_epochs
            self.rich_manager = RichManager(show_table, total_train_epochs)
            self.trainer.remove_callback(ProgressCallback)

        def _calculate_metric_scores(self) -> Dict[str, List[float]]:
            """
            Calculate final evaluation scores based on metrics and test cases.

            Returns:
                Dict[str, List[float]]: Metric scores for each test case.
            """
            test_results = execute_test_cases(
                test_cases=self.evaluation_dataset.test_cases,
                metrics=self.metrics,
            )
            scores = {}
            for test_result in test_results:
                for metric in test_result.metrics:
                    metric_name = str(metric.__name__)
                    metric_score = metric.score
                    scores.setdefault(metric_name, []).append(metric_score)

            scores = self._aggregate_scores(scores)
            return scores

        def _aggregate_scores(
            self, scores: Dict[str, List[float]]
        ) -> Dict[str, float]:
            """
            Aggregate metric scores using the specified method.

            Args:
                aggregation_method (str): Method for aggregating scores.
                scores (Dict[str, List[float]]): Metric scores for each test case.

            Returns:
                Dict[str, float]: Aggregated metric scores.
            """
            aggregation_functions = {
                "avg": lambda x: sum(x) / len(x),
                "max": max,
                "min": min,
            }
            if self.aggregation_method not in aggregation_functions:
                raise ValueError(
                    "Incorrect 'aggregation_method', only accepts ['avg', 'min, 'max']"
                )
            return {
                key: aggregation_functions[self.aggregation_method](value)
                for key, value in scores.items()
            }

        def on_epoch_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            """
            Event triggered at the begining of each training epoch.
            """
            self.epoch_counter += 1

        def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            """
            Event triggered at the end of each training epoch.
            """
            control.should_log = True
            self.rich_manager.change_spinner_text(
                self.task_descriptions["generating"]
            )
            test_cases = generate_test_cases(
                self.trainer.model,
                self.trainer.tokenizer,
                self.tokenizer_args,
                self.evaluation_dataset,
            )
            self.evaluation_dataset.test_cases = test_cases

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            """
            Event triggered after logging the last logs.
            """
            if (
                self.show_table
                and len(state.log_history) <= self.trainer.args.num_train_epochs
            ):
                self.rich_manager.advance_progress()

                self.rich_manager.change_spinner_text(
                    self.task_descriptions["evaluate"]
                )

                scores = self._calculate_metric_scores()
                self.deepeval_metric_history.append(scores)
                self.deepeval_metric_history[-1].update(state.log_history[-1])

                self.rich_manager.change_spinner_text(
                    self.task_descriptions["training"]
                )
                columns = self._generate_table()
                self.rich_manager.update(columns)

        def _generate_table(self):
            """
            Generates table, along with progress bars

            Returns:
                rich.Columns: contains table and 2 progress bars
            """
            column, table = self.rich_manager.create_column()
            order = get_column_order(self.deepeval_metric_history[-1])

            if self.show_table:
                for key in order:
                    table.add_column(key)

                for row in self.deepeval_metric_history:
                    table.add_row(*[str(row[value]) for value in order])

            return column

        def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            """
            Event triggered at the end of model training.
            """
            self.rich_manager.change_spinner_text(
                self.task_descriptions["training_end"]
            )
            self.rich_manager.stop()

        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            """
            Event triggered at the begining of model training.
            """
            self.rich_manager.start()
            self.rich_manager.change_spinner_text(
                self.task_descriptions["training"]
            )

except ImportError:

    class DeepEvalHuggingFaceCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The 'transformers' library is required to use the DeepEvalHuggingFaceCallback."
            )
