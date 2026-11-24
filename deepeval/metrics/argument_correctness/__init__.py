"""Deprecated: ``ArgumentCorrectnessMetric`` is now ``ArgumentRelevancyMetric``.

The metric judges whether tool-call arguments fit the user's input; it never
compares them with expected values, so "correctness" was misleading. To check
arguments against expected tool calls, use ``ToolCorrectnessMetric`` with
``evaluation_params=[ToolCallParams.INPUT_PARAMETERS]``.
"""

import warnings

from deepeval.metrics.argument_relevancy import (
    ArgumentRelevancyMetric,
    ArgumentRelevancyTemplate,
)

ArgumentCorrectnessTemplate = ArgumentRelevancyTemplate


class ArgumentCorrectnessMetric(ArgumentRelevancyMetric):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ArgumentCorrectnessMetric is deprecated and will be removed in a "
            "future release. Use ArgumentRelevancyMetric instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["ArgumentCorrectnessMetric", "ArgumentCorrectnessTemplate"]
