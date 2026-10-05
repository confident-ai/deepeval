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

        # Gather every parameter the constructor can accept from the whole MRO,
        # so configuration a base class ``__init__`` stored on the instance is
        # replayed onto the copy too.
        superclasses = metric_class.__mro__
        valid_params = set()
        for superclass in superclasses:
            valid_params.update(
                inspect.signature(superclass.__init__).parameters
            )
        valid_args = {key: args[key] for key in valid_params if key in args}

        target_params = inspect.signature(metric_class.__init__).parameters
        accepts_var_keyword = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in target_params.values()
        )
        # The leaf constructor accepts every collected kwarg (directly or via
        # ``**kwargs``): replay them unchanged, exactly as before.
        if accepts_var_keyword or valid_args.keys() <= set(target_params):
            copied_metrics.append(metric_class(**valid_args))
            continue

        # A subclass may deliberately narrow its ``__init__`` signature — the
        # documented way to pre-configure a built-in metric (e.g. a GEval
        # subclass that hard-codes ``name``/``criteria``). Replaying parent-only
        # kwargs into that constructor raised a ``TypeError`` that aborted the
        # whole async ``evaluate()`` run (#3037). Fall back to the params the
        # leaf constructor accepts; its ``__init__`` re-applies the pre-set
        # values itself, so no configuration is lost.
        narrowed_args = {key: args[key] for key in target_params if key in args}
        copied_metrics.append(metric_class(**narrowed_args))
    return copied_metrics
