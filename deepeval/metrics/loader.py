from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from .registry import metric_class_mapping
from .utils import ConfigLoader
from .base_metric import BaseMetric
from typing import List

class MetricsLoader:
    def __init__(self, config_path=None, metrics=None):
        """
        Initialize MetricsLoader instance

        Args:
            config_path (str, optional): Path to a YAML config file.
            metrics (list, optional): List of metrics to evaluate.

        Raises:
            ValueError: If neither config_path nor metrics are provided.
        """
        if config_path is None and metrics is None:
            raise ValueError("Either config_path or metrics must be provided")

        self.config_loader = None
        self.metrics = None

        if config_path is not None:
            self.config_loader = ConfigLoader(config_path)
            self.metrics = self.initialize_metrics()
        elif metrics is not None:
            self.metrics = metrics

        if self.config_loader is None:
            raise ValueError("Config file is not provided")
        if self.metrics is None:
            raise ValueError("Metrics are not provided")

    def initialize_metrics(
        self,
    ) -> dict:
        """
        Initialize metrics from config file.

        Initializes metrics for evaluation based on the configuration
        provided in the config file. The configuration is expected to be a dictionary
        where the keys are the names of the metrics and the values are dictionaries
        containing the configuration for the metric.

        Returns:
            dict: A dictionary containing the initialized metrics `{metric_name: metric object}`.
        """
        metrics_config = self.config_loader.get_metrics_config()
        metrics = {}
        for metric_name, config in metrics_config.items():
            # Map evaluation_params from config to LLMTestCaseParams
            evaluation_params = config.pop("evaluation_params", [])
            if not isinstance(evaluation_params, list):
                raise ValueError(
                    f"Invalid configuration for metric '{metric_name}'. "
                    f"'evaluation_params' must be a list. Check the metric registry for valid configuration."
                )
            # For handling multiple evaluation_params provided for some metrics (i.e. geval)
            mapped_params = []
            for param in evaluation_params:
                try:
                    # Convert the string param to the corresponding LLMTestCaseParams enum
                    mapped_param = getattr(LLMTestCaseParams, param.upper(), None)
                    if mapped_param is None:
                        raise ValueError(
                            f"Invalid evaluation param '{param}' for metric '{metric_name}'. "
                            f"Check the LLMTestCaseParams enum for valid values."
                        )
                    mapped_params.append(mapped_param)
                except AttributeError:
                    raise ValueError(
                        f"Invalid evaluation param '{param}' for metric '{metric_name}'. "
                        f"Check the LLMTestCaseParams enum for valid values."
                    )
            if mapped_params:
                config["evaluation_params"] = mapped_params
            if metric_name in metric_class_mapping:
                MetricClass = metric_class_mapping[metric_name]
                try:
                    metrics[metric_name] = MetricClass(**config)
                except TypeError:
                    raise ValueError(
                        f"Invalid configuration for metric '{metric_name}'. "
                        f"Check the metric registry for valid configuration."
                    )
            else:
                raise ValueError(f"No metric class found for '{metric_name}'. Check the metric registry.")

        return metrics

    def evaluate(
        self,
        test_case: LLMTestCase
    ) -> dict:
        """
        Evaluates the given test case using all the metrics in the metrics dictionary.

        Returns:
            dict[str, dict[str, Union[str, bool]]]: A dictionary containing the results of the evaluation for each metric.
        """
        results = {}
        for metric_name, metric in self.metrics.items():
            try:
                result = metric.measure(test_case)
                results[metric_name] = result
            except Exception as e:
                results[metric_name] = {
                    'error': str(e),
                    'success': False
                }
        return results

    def get_metrics_list(
        self,
    ) -> List[BaseMetric]:
        """
        Retrieves a list of metric objects from the MetricsEvaluator instance.

        Args:
            self (MetricsEvaluator): An instance of MetricsEvaluator.

        Returns:
            list: A list of metric objects.
        """
        return list(self.metrics.values())