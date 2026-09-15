"""copy_metrics must replay configuration onto a fresh metric instance even
when the metric subclasses a built-in with a narrower ``__init__``.

Regression test for #3037: reconstructing a copy used to collect candidate
kwargs from every class in the MRO but replay them into the leaf constructor,
so any subclass that deliberately narrows its ``__init__`` (the documented way
to pre-configure a built-in metric) crashed with a ``TypeError`` and aborted
the whole async ``evaluate()`` run before a single metric executed.
"""

import pytest

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import ExactMatchMetric, ToolPermissionMetric
from deepeval.metrics.utils import copy_metrics
from deepeval.test_case import LLMTestCase, ToolCall


class NoShellMetric(ToolPermissionMetric):
    """Reusable policy metric: a team-wide 'never call shell' gate."""

    def __init__(self, threshold: float = 1.0):
        super().__init__(denied_tools=["shell"], threshold=threshold)


class StrictSafeToolsMetric(ToolPermissionMetric):
    """Two-level narrowing: only read/write allowed, shell denied."""

    def __init__(self, threshold: float = 1.0):
        super().__init__(
            allowed_tools=["read", "write"],
            denied_tools=["shell"],
            threshold=threshold,
        )


_QUIET_DISPLAY = DisplayConfig(show_indicator=False, print_results=False)
_QUIET_ASYNC = AsyncConfig(run_async=False)


def _case() -> LLMTestCase:
    return LLMTestCase(
        input="hi",
        actual_output="hello",
        tools_called=[ToolCall(name="search")],
    )


# --- Default behaviour must not regress ----------------------------------


def test_copy_plain_metric_preserves_config():
    metric = ToolPermissionMetric(denied_tools=["shell"], threshold=0.9)
    copied = copy_metrics([metric])[0]

    assert copied is not metric
    assert copied.denied_tools == {"shell"}
    assert copied.threshold == 0.9


def test_copy_preserves_flaky_and_flags():
    metric = ExactMatchMetric(flaky=True, verbose_mode=False)
    copied = copy_metrics([metric])[0]

    assert copied.flaky is True
    assert copied.verbose_mode is False


def test_copies_are_fresh_instances_that_do_not_leak_score():
    metric = NoShellMetric()
    first, second = copy_metrics([metric, metric])

    first.measure(_case(), _show_indicator=False)
    assert first.score == 1.0
    assert second.score is None


# --- New capability: narrowed subclasses copy without crashing ------------


def test_copy_narrowed_subclass_does_not_raise():
    metric = NoShellMetric(threshold=0.8)
    copied = copy_metrics([metric])[0]

    assert type(copied) is NoShellMetric
    assert copied.denied_tools == {"shell"}
    assert copied.threshold == 0.8


def test_copy_narrowed_subclass_keeps_defaults():
    copied = copy_metrics([NoShellMetric()])[0]

    assert copied.threshold == 1.0
    assert copied.denied_tools == {"shell"}


def test_copy_narrowed_subclass_keeps_non_default_threshold():
    copied = copy_metrics([NoShellMetric(threshold=0.5)])[0]

    assert copied.threshold == 0.5


def test_copy_narrowed_subclass_multiple_config_values():
    copied = copy_metrics([StrictSafeToolsMetric(threshold=0.6)])[0]

    assert type(copied) is StrictSafeToolsMetric
    assert copied.allowed_tools == {"read", "write"}
    assert copied.denied_tools == {"shell"}
    assert copied.threshold == 0.6


def test_evaluate_async_runs_narrowed_subclass_end_to_end():
    result = evaluate(test_cases=[_case()], metrics=[NoShellMetric()])

    assert len(result.test_results) == 1
    assert result.test_results[0].metrics_data[0].score == 1.0


def test_evaluate_sync_runs_narrowed_subclass_end_to_end():
    result = evaluate(
        test_cases=[_case()],
        metrics=[NoShellMetric()],
        display_config=_QUIET_DISPLAY,
        async_config=_QUIET_ASYNC,
    )

    assert len(result.test_results) == 1
    assert result.test_results[0].metrics_data[0].score == 1.0
