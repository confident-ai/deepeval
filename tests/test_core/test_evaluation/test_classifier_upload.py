"""Classifications are uploaded with their test case: on the API test case,
folded into its pass/fail verdict and evaluation cost, and kept in the test
run even when no metric ran."""

import pytest

from deepeval import evaluate
from deepeval.classifiers.base_classifier import BaseClassifier, Label
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
from deepeval.test_run import global_test_run_manager
from deepeval.test_run.api import Classification
from deepeval.test_run.test_run import (
    TestRun,
    TestRunManager,
    TestRunResultDisplay,
)


class Greeting(BaseClassifier):
    name = "Greeting"
    labels = [Label(name="greeting"), Label(name="other")]

    def classify(self, test_case, **kwargs):
        self.label = "greeting"
        self.reason = "says hello"
        self.evaluation_cost = 0.25
        return self.label

    async def a_classify(self, test_case, **kwargs):
        return self.classify(test_case)


class PassingMetric(BaseMetric):
    threshold = 1

    @property
    def __name__(self):
        return "Existing metric"

    def measure(self, test_case, **kwargs):
        self.score = 1
        self.evaluation_cost = 0.5
        return self.score

    async def a_measure(self, test_case, **kwargs):
        return self.measure(test_case)


@pytest.fixture(autouse=True)
def captured_test_run(monkeypatch):
    global_test_run_manager.reset()
    monkeypatch.setattr(
        global_test_run_manager,
        "wrap_up_test_run",
        lambda *a, **k: (None, None),
    )
    yield
    global_test_run_manager.reset()


def run(cases, run_async, **kwargs):
    return evaluate(
        cases,
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        cache_config=CacheConfig(use_cache=False, write_cache=False),
        **kwargs,
    ).test_results


def uploaded_test_cases():
    body = global_test_run_manager.get_test_run().model_dump(
        by_alias=True, exclude_none=True
    )
    return body.get("testCases", []) + body.get("conversationalTestCases", [])


@pytest.mark.parametrize("run_async", [False, True])
def test_failed_classification_fails_the_uploaded_test_case(run_async):
    case = LLMTestCase(
        input="hello",
        actual_output="hello",
        expected_labels={"Greeting": "other"},
    )

    [result] = run(
        [case], run_async, metrics=[PassingMetric()], classifiers=[Greeting()]
    )

    [uploaded] = uploaded_test_cases()
    assert uploaded["success"] is False
    assert result.success is False
    assert uploaded["classifications"] == [
        {
            "name": "Greeting",
            "label": "greeting",
            "expectedLabel": "other",
            "success": False,
            "reason": "says hello",
            "evidence": [],
            "evaluationCost": 0.25,
        }
    ]
    assert uploaded["evaluationCost"] == pytest.approx(0.75)


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("conversational", [False, True])
def test_classifier_only_test_case_is_uploaded(run_async, conversational):
    case = (
        ConversationalTestCase(
            turns=[Turn(role="user", content="hello")],
            expected_labels={"Greeting": "greeting"},
        )
        if conversational
        else LLMTestCase(
            input="hello",
            actual_output="hello",
            expected_labels={"Greeting": "greeting"},
        )
    )

    run([case], run_async, classifiers=[Greeting()])

    [uploaded] = uploaded_test_cases()
    assert uploaded["success"] is True
    assert [c["label"] for c in uploaded["classifications"]] == ["greeting"]


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("conversational", [False, True])
def test_classification_without_expected_label_uploads_no_verdict(
    run_async, conversational
):
    case = (
        ConversationalTestCase(turns=[Turn(role="user", content="hello")])
        if conversational
        else LLMTestCase(input="hello", actual_output="hello")
    )

    [result] = run([case], run_async, classifiers=[Greeting()])

    [uploaded] = uploaded_test_cases()
    assert "success" not in uploaded
    assert result.success is None
    assert "success" not in uploaded["classifications"][0]


def test_classifier_only_run_has_no_metrics_data():
    test_run = TestRun(
        testCases=[
            {
                "name": "case",
                "input": "hello",
                "metricsData": [],
                "classifications": [
                    Classification(name="Greeting", label="greeting")
                ],
            }
        ]
    )

    assert test_run.has_metrics_data() is False


def test_failing_display_skips_test_cases_without_a_verdict():
    test_run = TestRun(
        testCases=[{"name": "case", "input": "hello", "metricsData": []}]
    )
    [test_case] = test_run.test_cases
    test_case.success = None

    assert TestRunManager._should_skip_test_case(
        test_case, TestRunResultDisplay.FAILING
    )
    assert TestRunManager._should_skip_test_case(
        test_case, TestRunResultDisplay.PASSING
    )


def test_multi_label_classifier_counts_once_in_the_results_table():
    classifications = [
        Classification(name="Topic", label=label, success=False)
        for label in ("pricing", "refund")
    ] + [Classification(name="Greeting", label="greeting", success=True)]

    assert TestRunManager._count_metric_results(
        TestRunManager._classifier_verdicts(classifications)
    ) == (1, 1, 0, 0)
