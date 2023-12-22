from typing import Union, List, Dict

from deepeval.metrics import BaseMetric
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import execute_test

from transformers.trainer_callback import TrainerCallback


# TODO:
#   1. Dataset has to be created dynamically
#   2. add score to default dict returned by on_epoch_end
#   3. Make code more presentable

class DeepEvalCallback(TrainerCallback):
    """
    A Transformers TrainerCallback that logs various Language Model (LM) evaluation metrics to DeepEval.
    """

    def __init__(
        self,
        metrics: Union[BaseMetric, List[BaseMetric]] = None,
        evaluation_dataset: EvaluationDataset = None,
        tokenizer_args: Dict = {},
        aggregation_method: str = "average",
    ):
        """
        Initialize the DeepEvalCallback.

        Args:
            metrics (Union[BaseMetric, List[BaseMetric]], optional): Evaluation metrics to calculate.
                Defaults to None.
            evaluation_dataset (EvaluationDataset, optional): Dataset for evaluation. Defaults to None.
            tokenizer_args (Dict, optional): Additional arguments for tokenizer. Defaults to {}.
            aggregation_method (str, optional): Aggregation method for metric scores ("average", "max", "min").
                Defaults to "average".
        """
        super().__init__()
        self.metrics = metrics
        self.evaluation_dataset = evaluation_dataset
        self.tokenizer_args = tokenizer_args
        self.aggregation_method = aggregation_method

    def _calculate_scores(self) -> Dict[str, List[float]]:
        """
        Calculate scores for each evaluation metric.

        Returns:
            Dict[str, List[float]]: Dictionary containing metric names and corresponding scores.
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

        return scores

    def _aggregate_scores(
        self, aggregation_method: str, 
        scores: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Aggregate metric scores based on the specified method.

        Args:
            aggregation_method (str): Aggregation method ("average", "max", "min").
            scores (Dict[str, List[float]]): Dictionary containing metric names and scores.

        Returns:
            Dict[str, float]: Dictionary containing aggregated metric names and scores.
        """
        if aggregation_method in ["average", "avg"]:
            scores = {key: (sum(value) / len(value)) for key, value in scores.items()}
        elif aggregation_method == "max":
            scores = {key: (max(value)) for key, value in scores.items()}
        elif aggregation_method == "min":
            scores = {key: (min(value)) for key, value in scores.items()}
        else:
            raise ValueError("Incorrect 'aggregation_method' passed, only accepts ['avg', 'min, 'max']")
        return scores

    def on_epoch_end(self, args, state, control, model, tokenizer, **kwargs):
        """
        Called at the end of each training epoch.

        Args:
            args: Training arguments.
            state: Training state.
            control: Training control.
            model: The current model.
            tokenizer: Tokenizer used for evaluation.
            kwargs: Additional keyword arguments.
        """
        scores = self._calculate_scores()
        scores = self._aggregate_scores(self.aggregation_method, scores)
        print(scores)

    def on_train_end(self, args, state, control, model, tokenizer, **kwargs):
        """
        Called at the end of the training process.

        Args:
            args: Training arguments.
            state: Training state.
            control: Training control.
            model: The final model.
            tokenizer: Tokenizer used for evaluation.
            kwargs: Additional keyword arguments.
        """
        print("---------TRAIN END---------")



# REWRITE THIS IN A BETTER PROFESSIONAL WAY, MAKE DOCS AND ERRORS MORE PROFESSIONAL, MAKE CODE BETTER, OPTIMIZE THE CODE AS MUCH AS YOU CAN TO LOOK BETTER