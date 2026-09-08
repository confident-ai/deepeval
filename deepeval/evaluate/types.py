from typing import Optional, List, Union, Dict
from dataclasses import dataclass, replace
from pydantic import BaseModel

from deepeval.test_run.api import MetricData, TurnApi
from deepeval.test_case import MLLMImage
from deepeval.test_run import TestRun


@dataclass
class TestResult:
    """Returned from run_test"""

    __test__ = False
    name: str
    success: bool
    metrics_data: Union[List[MetricData], None]
    conversational: bool
    index: Optional[int] = None
    multimodal: Optional[bool] = None
    input: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    actual_output: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    turns: Optional[List[TurnApi]] = None
    metadata: Optional[Dict] = None


def _recalculate_metric_success(
    metric_data: MetricData, threshold: float
) -> bool:
    """Recompute a single metric's pass/fail against a new threshold.

    Mirrors the pass/fail semantics used during evaluation: a metric that
    errored or has no score fails, otherwise it passes when its score meets
    the threshold.
    """
    if metric_data.error is not None:
        return False
    if metric_data.score is None:
        return False
    return metric_data.score >= threshold


def _recalculate_test_result(
    test_result: TestResult, thresholds: Dict[str, float]
) -> TestResult:
    """Return a copy of ``test_result`` with statuses recomputed.

    Metrics whose name is absent from ``thresholds`` keep their original
    threshold and status. The original ``test_result`` is not mutated.
    """
    if not test_result.metrics_data:
        return replace(test_result)

    recalculated_metrics_data: List[MetricData] = []
    for metric_data in test_result.metrics_data:
        if metric_data.name in thresholds:
            new_threshold = thresholds[metric_data.name]
            recalculated_metrics_data.append(
                metric_data.model_copy(
                    update={
                        "threshold": new_threshold,
                        "success": _recalculate_metric_success(
                            metric_data, new_threshold
                        ),
                    }
                )
            )
        else:
            recalculated_metrics_data.append(metric_data.model_copy())

    success = all(
        metric_data.success for metric_data in recalculated_metrics_data
    )
    return replace(
        test_result,
        metrics_data=recalculated_metrics_data,
        success=success,
    )


class EvaluationResult(BaseModel):
    test_results: List[TestResult]
    confident_link: Optional[str]
    test_run_id: Optional[str]

    def recalculate_results(
        self,
        thresholds: Dict[str, float],
        print_results: bool = True,
    ) -> "EvaluationResult":
        """Recompute pass/fail statuses and pass rates using new thresholds.

        Reuses the scores already produced by ``evaluate`` to recompute each
        metric's pass/fail status and the overall pass/fail of every test case
        against new per-metric thresholds. No LLM evaluations are re-run, and
        the original ``EvaluationResult`` (and its scores and reasons) is left
        untouched: a new ``EvaluationResult`` carrying the recalculated
        statuses is returned.

        :param thresholds: Mapping of metric name to the new threshold to
            apply. Metrics whose name is not present keep their original
            threshold and status.
        :param print_results: When ``True`` (default), print the recalculated
            overall metric pass rates.
        :return: A new ``EvaluationResult`` with recomputed statuses.
        """
        recalculated = EvaluationResult(
            test_results=[
                _recalculate_test_result(test_result, thresholds)
                for test_result in self.test_results
            ],
            confident_link=self.confident_link,
            test_run_id=self.test_run_id,
        )
        if print_results:
            # Imported lazily to avoid a circular import at module load time.
            from deepeval.evaluate.utils import aggregate_metric_pass_rates

            aggregate_metric_pass_rates(recalculated.test_results)
        return recalculated


class PostExperimentRequest(BaseModel):
    testRuns: List[TestRun]
    name: Optional[str]
