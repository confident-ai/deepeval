import pytest
from deepeval.metrics.contextual_relevancy.schema import Verdicts
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric
from deepeval import assert_test
from tests.custom_judge import CustomJudge

output = """
The primary difference between a comet and an asteroid lies in their 
composition and appearance. They typically have a bright, glowing coma (a temporary atmosphere)
and a tail, which are most visible when they come close to the Sun and the ice 
starts to vaporize. Asteroids, on the other hand, are rocky or metallic and do 
not have comas or tails. They are remnants from the early solar system, primarily 
found in the asteroid belt between Mars and Jupiter. Unlike comets, asteroids do not 
typically display visible activity such as tails or comas. Comets are loved by rabbits all around the world.
"""

one = """Comets and asteroids are both celestial bodies found in our solar system but differ in composition and behavior. Comets, made up of ice, dust, and small rocky particles, develop glowing comas and tails when near the Sun. In contrast, asteroids are primarily rocky or metallic and are mostly found in the asteroid belt between Mars and Jupiter. Rabbits are found in the wild."""

two = """The physical characteristics and orbital paths of comets and asteroids vary significantly. Comets often have highly elliptical orbits, taking them close to the Sun and then far into the outer solar system. Their icy composition leads to distinctive features like tails and comas. Lions eat rabbits. Asteroids, conversely, have more circular orbits and lack these visible features, being composed mostly of rock and metal."""

three = """Understanding comets and asteroids is crucial in studying the solar system's formation and evolution. Comets, which are remnants from the outer solar system, can provide insights into its icy and volatile components. Asteroids, primarily remnants of the early solar system's formation, offer clues about the materials that didn't form into planets, mostly located in the asteroid belt."""


@pytest.mark.skip(reason="openai is expensive")
def test_contextual_relevancy():
    test_case = LLMTestCase(
        input="What is the primary difference between a comet and an asteroid?",
        actual_output=output,
        expected_output=output,
        retrieval_context=[one, two, three],
    )
    metric = ContextualRelevancyMetric()
    assert_test(test_case, [metric])


def test_verdict_schema():

    judge = CustomJudge("mock")
    schema = Verdicts
    answer = (
        '{\n"verdicts": [\n{\n"verdict": "yes"\n},\n{\n    "verdict": "no",\n    "reason": "blah blah"\n},'
        '\n{\n    "verdict": "yes",\n    "reason":null \n}\n]\n}'
    )
    res: Verdicts = judge.generate(answer, schema=schema)
