import inspect
from typing import Callable, Union

from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase


def _build_measure_kwargs(func: Callable) -> dict:
    params = inspect.signature(func).parameters
    kwargs = {}
    for key in ("_show_indicator", "_in_component", "_log_metric_to_confident"):
        if key in params:
            kwargs[key] = False
    return kwargs


def _measure_no_indicator(
    metric: Union[BaseMetric, BaseConversationalMetric],
    test_case: Union[LLMTestCase, ConversationalTestCase],
):
    kwargs = _build_measure_kwargs(metric.measure)
    return metric.measure(test_case, **kwargs)


async def _a_measure_no_indicator(
    metric: Union[BaseMetric, BaseConversationalMetric],
    test_case: Union[LLMTestCase, ConversationalTestCase],
):
    kwargs = _build_measure_kwargs(metric.a_measure)
    return await metric.a_measure(test_case, **kwargs)
