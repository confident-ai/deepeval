import pytest
from deepeval.metrics.hallucination.schema import Verdicts
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
from deepeval import assert_test
from tests.custom_judge import CustomJudge


@pytest.mark.skip(reason="openai is expensive")
def test_hallucination_metric():
    metric = HallucinationMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="A blond drinking water in public.",
        context=[
            "A man with blond-hair, and a brown shirt drinking out of a public water fountain."
        ],
        cost=0.4,
        latency=2,
    )
    assert_test(test_case, [metric])


@pytest.mark.skip(reason="openai is expensive")
def test_hallucination_metric_2():
    metric = HallucinationMetric(threshold=0.6)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="Python is a programming language.",
        context=["Python is NOT a programming language."],
        cost=1,
        latency=0.2,
    )
    assert_test(test_case, [metric])


@pytest.mark.skip(reason="openai is expensive")
def test_hallucination_metric_3():
    metric = HallucinationMetric(threshold=0.6)
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="Python is a programming language.",
        context=["Python is a snake.", "Pythons like to lurk in the forests."],
        cost=0.1,
        latency=13.0,
    )
    assert_test(test_case, [metric])


def test_verdict_schema():

    judge = CustomJudge("mock")
    schema = Verdicts
    answer = (
        '{\n"verdicts": [\n'
        '{\n"verdict": "yes",\n"reason": ""\n},\n'
        '{\n"verdict": "no",\n"reason": "blah blah"\n},\n'
        '{\n"verdict": "yes",\n"reason": ""\n}\n'
        "]\n}"
    )
    res: Verdicts = judge.generate(answer, schema=schema)
