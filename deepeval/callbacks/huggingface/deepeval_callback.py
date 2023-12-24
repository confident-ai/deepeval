from typing import Union, List, Dict

from rich.console import Console
from rich.table import Table
from rich.live import Live
from transformers import TrainerCallback, \
    ProgressCallback, Trainer, \
    TrainingArguments, TrainerState, TrainerControl
    
from deepeval.metrics import BaseMetric
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import execute_test


class DeepEvalCallback(TrainerCallback):
    """
    Custom callback for deep evaluation during model training.

    Args:
        metrics (Union[BaseMetric, List[BaseMetric]]): Evaluation metrics.
        evaluation_dataset (EvaluationDataset): Dataset for evaluation.
        tokenizer_args (Dict): Arguments for the tokenizer.
        aggregation_method (str): Method for aggregating metric scores.
        trainer (Trainer): Model trainer.
    """

    def __init__(
        self,
        metrics: Union[BaseMetric, List[BaseMetric]] = None,
        evaluation_dataset: EvaluationDataset = None,
        tokenizer_args: Dict = None,
        aggregation_method: str = "avg",
        trainer: Trainer = None
    ) -> None:
        super().__init__()
        self.metrics = metrics
        self.evaluation_dataset = evaluation_dataset
        self.tokenizer_args = tokenizer_args
        self.aggregation_method = aggregation_method
        self.trainer = trainer
        
        self.epoch_counter = 0
        self.log_history = []
        self._initiate_rich_console()
        
    def _initiate_rich_console(self) -> None:
        """
        Initiate rich console for progress tracking.
        """
        console = Console()
        self.live = Live(auto_refresh=True, console=console)
        self.trainer.remove_callback(ProgressCallback)

    def _calculate_metric_scores(self) -> Dict[str, List[float]]:
        """
        Calculate final evaluation scores based on metrics and test cases.

        Returns:
            Dict[str, List[float]]: Metric scores for each test case.
        """
        test_results = execute_test(
            test_cases=self.evaluation_dataset.test_cases,
            metrics=self.metrics
        )
        scores = {}
        for test in test_results:
            for metric in test.metrics:
                metric_name = str(metric.__name__).lower().replace(" ", "_")
                metric_score = metric.score
                scores.setdefault(metric_name, []).append(metric_score)
                
        scores = self._aggregate_scores(scores)
        return scores

    def _aggregate_scores(self,
        scores: Dict[str, List[float]]
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
            raise ValueError("Incorrect 'aggregation_method', only accepts ['avg', 'min, 'max']")
        return {
            key: aggregation_functions[self.aggregation_method](value) \
                for key, value in scores.items()
        }

    def on_epoch_end(self, 
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event triggered at the end of each training epoch.
        """
        self.epoch_counter += 1
        scores = self._calculate_metric_scores()
        self.log_history.append(scores)
        control.should_log = True
        
        return control

    def on_log(self, 
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event triggered after logging the last logs.
        """
        if not control.should_training_stop:
            state.log_history[-1].update(self.log_history[-1])
            log_history = state.log_history

            def generate_table():
                new_table = Table()
                cols = log_history[-1].keys()
                for key in cols:
                    new_table.add_column(key)
                for row in log_history:
                    new_table.add_row(*[str(value) for value in row.values()])
                return new_table

            with self.live:
                self.live.console.clear()
                self.live.update(generate_table(), refresh=True)
        else:
            pass

    def on_train_end(self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event triggered at the end of model training.
        """
        print("---------TRAIN END---------")
