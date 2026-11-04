pytest_plugins = ["pytester"]


def test_plain_pytest_renders_results_table(pytester):
    pytester.makepyfile(
        test_eval="""
import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["COLUMNS"] = "200"

from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class AlwaysPassMetric(BaseMetric):
    threshold = 0.5

    def __init__(self):
        self.score = None

    def measure(self, test_case):
        self.score = 1.0
        return self.score

    async def a_measure(self, test_case):
        return self.measure(test_case)

    @property
    def __name__(self):
        return "AlwaysPassMetric"


def test_evaluation():
    evaluate(
        test_cases=[LLMTestCase(input="hello", actual_output="hello")],
        metrics=[AlwaysPassMetric()],
        display_config=DisplayConfig(show_indicator=False, print_results=False, inspect_after_run=False),
    )
"""
    )
    result = pytester.runpytest()
    # The table title and a metric name must be visible after the run.
    assert "Test Results" in result.stdout.str()
    assert "AlwaysPassMetric" in result.stdout.str()


def test_unrelated_suite_prints_no_table(pytester):
    pytester.makepyfile(test_plain="""def test_ok():\n    assert True\n""")
    result = pytester.runpytest()
    assert "Test Results" not in result.stdout.str()


def test_assert_test_session_renders_results_table(pytester):
    pytester.makepyfile(
        test_assert="""
import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["COLUMNS"] = "200"

from deepeval.evaluate import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class AlwaysPassMetric(BaseMetric):
    threshold = 0.5

    def __init__(self):
        self.score = None

    def measure(self, test_case):
        self.score = 1.0
        return self.score

    async def a_measure(self, test_case):
        return self.measure(test_case)

    @property
    def __name__(self):
        return "AlwaysPassMetric"


def test_evaluation():
    assert_test(
        test_case=LLMTestCase(input="hello", actual_output="hello"),
        metrics=[AlwaysPassMetric()],
    )
"""
    )
    result = pytester.runpytest()
    assert "Test Results" in result.stdout.str()
    assert "AlwaysPassMetric" in result.stdout.str()


def test_failing_assert_test_still_renders_results_table(pytester):
    pytester.makepyfile(
        test_assert="""
import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["COLUMNS"] = "200"

from deepeval.evaluate import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class AlwaysFailMetric(BaseMetric):
    threshold = 0.5

    def __init__(self):
        self.score = None

    def measure(self, test_case):
        self.score = 0.0
        return self.score

    async def a_measure(self, test_case):
        return self.measure(test_case)

    @property
    def __name__(self):
        return "AlwaysFailMetric"


def test_evaluation():
    assert_test(
        test_case=LLMTestCase(input="hello", actual_output="hello"),
        metrics=[AlwaysFailMetric()],
    )
"""
    )
    result = pytester.runpytest()
    # The inner test fails (metric score 0.0 < threshold), but the results
    # table must still be rendered after the session.
    assert result.ret != 0
    assert "Test Results" in result.stdout.str()
    assert "AlwaysFailMetric" in result.stdout.str()
