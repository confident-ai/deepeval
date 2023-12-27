from typing import Union, List, Dict

from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.live import Live
from transformers import TrainerCallback, \
    ProgressCallback, Trainer, \
    TrainingArguments, TrainerState, TrainerControl
    
from deepeval.metrics import BaseMetric
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import execute_test
from deepeval.progress_context import progress_context


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
        trainer: Trainer = None,
        show_table: bool = False,
        show_table_every: int = 1
    ) -> None:
        super().__init__()
        
        self.show_table = show_table
        self.show_table_every = show_table_every
        self.metrics = metrics
        self.evaluation_dataset = evaluation_dataset
        self.tokenizer_args = tokenizer_args
        self.aggregation_method = aggregation_method
        self.trainer = trainer
        
        self.epoch_counter = 0
        self.deepeval_metric_history = []
        self._initiate_rich_console()

    def _initiate_rich_console(self) -> None:
        """
        Initiate rich console for progress tracking.
        """
        if self.show_table:
            self.console = Console()
            self.live = Live(auto_refresh=True, console=self.console)
            self.live.start()
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
        
    def on_epoch_begin(self, 
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event triggered at the begining of each training epoch.
        """
        self.epoch_counter += 1

    def on_epoch_end(self, 
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event triggered at the end of each training epoch.
        """
        self.progress.update(1)
        control.should_log = True


    def on_log(self, 
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event triggered after logging the last logs.
        """
        if (
            self.show_table 
            and (self.epoch_counter % self.show_table_every == 0) 
            and len(state.log_history) <= self.trainer.args.num_train_epochs
        ):
            with progress_context("Evaluating testcases..."):
                scores = self._calculate_metric_scores()
                self.deepeval_metric_history.append(scores)
                self.deepeval_metric_history[-1].update(state.log_history[-1])

            def generate_table():
                new_table = Table()
                for key in self.deepeval_metric_history[-1].keys():
                    new_table.add_column(key)
                for row in self.deepeval_metric_history:
                    new_table.add_row(*[str(value) for value in row.values()])
                return new_table
            self.live.update(generate_table(), refresh=True)

    def on_train_end(self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event triggered at the end of model training.
        """
        self.progress.close()
        
    def on_train_begin(self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event triggered at the begining of model training.
        """
        self.progress = tqdm(
            total=self.trainer.args.num_train_epochs, 
            desc="Epochs"
        )
        
        
