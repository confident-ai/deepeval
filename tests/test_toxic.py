import pytest
from deepeval.metrics.toxicity.schema import Verdicts
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToxicityMetric
from deepeval import assert_test
from tests.custom_judge import CustomJudge

output = """
This implementation seems a bit convoluted. 
I know you're new to the team, but there are probably more elegant ways to achieve this functionality. 
Have you considered looking at the patterns in Module X for inspiration? 
Just trying to help you learn our best practices.
"""


@pytest.mark.skip(reason="openai is expensive")
def test_bias():
    test_case = LLMTestCase(
        input="What is the primary difference between a comet and an asteroid?",
        actual_output=output,
    )
    metric = ToxicityMetric()
    assert_test(test_case, [metric])


def test_verdict_schema():

    judge = CustomJudge("mock")
    schema = Verdicts
    answer = (
        '{\n"verdicts": [\n{\n"verdict": "yes"\n},\n{\n    "verdict": "no",\n    "reason": "blah blah"\n},'
        '\n{\n    "verdict": "yes",\n    "reason":null \n}\n]\n}'
    )
    res: Verdicts = judge.generate(answer, schema=schema)
