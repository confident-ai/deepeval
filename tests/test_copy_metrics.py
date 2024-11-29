import pytest

from deepeval.metrics import GEval
from deepeval.metrics.utils import copy_metrics
from deepeval.test_case import LLMTestCaseParams
from deepeval.models.gpt_model import GPTModel


class DummyMetric(GEval):
    def __init__(self, **kwargs):
        kwargs["evaluation_params"] = [
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.INPUT,
        ]
        kwargs["criteria"] = "All answers are good"
        if "name" not in kwargs.keys():
            kwargs["name"] = "default_config_name"
        super().__init__(**kwargs)


def test_copy_metrics():
    # Different than the default, 'gpt-4o'
    metric_before = DummyMetric(
        model="gpt-4o-mini",
    )
    metric_after = copy_metrics([metric_before])
    vars_before = vars(metric_before)
    vars_after = vars(metric_after[0])
    for key_before, value_before in vars_before.items():
        value_after = vars_after[key_before]
        if isinstance(value_before, GPTModel):
            assert value_before.model_name == value_after.model_name
        else:
            assert value_before == value_after
        assert key_before in vars_after.keys()
