import inspect
import warnings
from typing import List, Set, Union

from deepeval.metrics.base_metric import (
    BaseMetric,
    BaseConversationalMetric,
)

# Metrics whose score direction was inverted; each warns once per process so a
# run that builds the metric per test case doesn't repeat itself.
_score_direction_warned: Set[str] = set()


def warn_score_direction_flipped(metric_name: str) -> None:
    if metric_name in _score_direction_warned:
        return
    _score_direction_warned.add(metric_name)
    warnings.warn(
        f"'{metric_name}' now scores in the same direction as every other "
        "deepeval metric: 1 is a pass, 0 is a failure, and 'threshold' is the "
        "MINIMUM passing score. It previously scored the proportion of "
        "violations, where 'threshold' was a maximum. Review any 'threshold' "
        "you pass and any code reading '.score' - a threshold of 0.2 that "
        "used to mean 'at most 20% violations' should now be 0.8. This notice "
        "will be removed in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


def check_at_least_one_metric_has_threshold(
    metrics: List[Union[BaseMetric, BaseConversationalMetric]],
):
    # Flaky metrics don't count: they must not decide a test case's
    # pass/fail status, so every test case needs at least one non-flaky
    # metric with a threshold to guarantee it gets a verdict.
    if not any(
        metric.threshold is not None and not metric.flaky for metric in metrics
    ):
        raise ValueError(
            "You must provide at least one non-flaky metric with a "
            "'threshold', otherwise test cases can never pass or fail."
        )


def copy_metrics(
    metrics: List[Union[BaseMetric, BaseConversationalMetric]],
) -> List[Union[BaseMetric, BaseConversationalMetric]]:
    copied_metrics = []
    for metric in metrics:
        metric_class = type(metric)
        args = vars(metric)

        superclasses = metric_class.__mro__

        valid_params = []

        for superclass in superclasses:
            signature = inspect.signature(superclass.__init__)
            superclass_params = signature.parameters.keys()
            valid_params.extend(superclass_params)
        valid_params = set(valid_params)
        valid_args = {key: args[key] for key in valid_params if key in args}

        copied_metrics.append(metric_class(**valid_args))
    return copied_metrics
