"""Function for running test
"""
import os
from typing import List, Optional
from .client import Client
from .test_case import LLMTestCase
from .metrics import Metric
from .get_api_key import _get_api_key, _get_implementation_name
from .constants import IMPLEMENTATION_ID_ENV, LOG_TO_SERVER_ENV


def _is_api_key_set():
    result = _get_api_key()
    # if result == "" or result is None:
    #     warnings.warn(
    #         """API key is not set - you won't be able to log to the DeepEval dashboard. Please set it by running `deepeval login`"""
    #     )
    if result == "" or result is None:
        return False
    return True


def _is_send_okay():
    # DOing this until the API endpoint is fixed
    return _is_api_key_set() and os.getenv(LOG_TO_SERVER_ENV) != "Y"


def _get_init_values(metric: Metric):
    # We use this method for sending useful metadata
    init_values = {
        param: getattr(metric, param)
        for param in vars(metric)
        if isinstance(getattr(metric, param), (str, int, float))
    }
    return init_values


def _send_to_server(
    self,
    success: bool,
    metric_score: float,
    metric_name: str,
    query: str = "-",
    output: str = "-",
    expected_output: str = "-",
    metadata: Optional[dict] = None,
    context: str = "-",
    **kwargs,
):
    if _is_send_okay():
        api_key = _get_api_key()
        client = Client(api_key=api_key)
        implementation_name = _get_implementation_name()
        # implementation_id = os.getenv(IMPLEMENTATION_ID_ENV, "")
        # if implementation_id != "":
        implementation_id = client.get_implementation_id_by_name(
            implementation_name
        )
        os.environ[IMPLEMENTATION_ID_ENV] = implementation_id
        datapoint_id = client.add_golden(
            query=query, expected_output=expected_output, context=context
        )

        metric_metadata: dict = _get_init_values()
        if metadata:
            metric_metadata.update(metadata)

        return client.add_test_case(
            metric_score=float(metric_score),
            metric_name=metric_name,
            actual_output=output,
            query=query,
            implementation_id=implementation_id,
            metrics_metadata=metric_metadata,
            success=bool(success),
            datapoint_id=datapoint_id["id"],
        )


def log(
    success: bool = True,
    score: float = 1e-10,
    metric_name: str = "-",
    query: str = "-",
    output: str = "-",
    expected_output: str = "-",
    metadata: Optional[dict] = None,
    context: str = "-",
):
    """Log to the server.

    Parameters
    - query: What was asked to the model. This can also be context.
    - output: The LLM output.
    - expected_output: The output that's expected.
    """
    if _is_send_okay():
        _send_to_server(
            metric_score=score,
            metric_name=metric_name,
            query=query,
            output=output,
            expected_output=expected_output,
            success=success,
            metadata=metadata,
            context=context,
        )


def run_test(test_case: LLMTestCase, metrics: List[Metric]):
    """
    Args:
        test_case: Test case to run
        metric: Metric to run

    Example:
        >>> from deepeval.metrics.facutal_consistency import FactualConsistencyMetric
        >>> from deepeval.test_case import LLMTestCase
        >>> from deepeval.run_test import run_test
        >>> metric = FactualConsistencyMetric()
        >>> test_case = LLMTestCase(
            ...     query="What is the capital of France?",
            ...     output="Paris",
            ...     expected_output="Paris",
            ...     context="Geography",
            ... )
        >>> run_test(test_case, metric)
    """
    for metric in metrics:
        score = metric.measure(test_case)
        success = metric.is_successful()
        log(
            success=success,
            score=score,
            metric_name=metric.__name__,
            query=test_case.query if test_case.query else "-",
            output=test_case.output if test_case.output else "-",
            expected_output=test_case.expected_output
            if test_case.expected_output
            else "-",
            context=test_case.context,
        )

        assert (
            metric.is_successful()
        ), f"{metric.__name__} failed. Score: {score}."
