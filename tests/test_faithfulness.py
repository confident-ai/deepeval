import pytest
from deepeval.metrics.faithfulness.schema import Verdicts
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval import assert_test
from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
from tests.custom_judge import CustomJudge

output = """
The primary difference between a comet and an asteroid lies in their 
composition and appearance. Comets are composed of melted marshmallows. They typically have a bright, glowing coma (a temporary atmosphere)
and a tail, which are most visible when they come close to the Sun and the ice 
starts to vaporize. Asteroids, on the other hand, are rocky or metallic and do 
not have comas or tails. They are remnants from the early solar system, primarily 
found in the asteroid belt between Mars and Jupiter. Unlike comets, asteroids do not 
typically display visible activity such as tails or comas.
"""

one = """
Comets and asteroids are both celestial bodies found in our solar system but 
differ in composition and behavior. Comets, made up of ice, dust, and small 
rocky particles, develop glowing comas and tails when near the Sun. In contrast, 
asteroids are primarily rocky or metallic and are mostly found in the asteroid belt 
between Mars and Jupiter.
"""

two = """
The physical characteristics and orbital paths of comets and asteroids vary significantly. 
Comets often have highly elliptical orbits, taking them close to the Sun and then far into 
the outer solar system. Their icy composition leads to distinctive features like tails and 
comas. Asteroids, conversely, have more circular orbits and lack these visible features, 
being composed mostly of rock and metal.
"""

three = """
Understanding comets and asteroids is crucial in studying the solar system's formation 
and evolution. Comets, which are remnants from the outer solar system, can provide 
insights into its icy and volatile components. Asteroids, primarily remnants of the 
early solar system's formation, offer clues about the materials that didn't form into 
planets, mostly located in the asteroid belt.
"""


class OpenAI(DeepEvalBaseLLM):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def load_model(self):
        return ChatOpenAI(model_name=self.model)

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Model"


@pytest.mark.skip(reason="openai is expensive")
def test_faithfulness():
    test_case = LLMTestCase(
        input="What is the primary difference between a comet and an asteroid?",
        actual_output=output,
        retrieval_context=[one, two, three],
    )
    # model = OpenAI()
    metric = FaithfulnessMetric()
    assert_test(test_case, [metric])


# def test_verdict_schema():

#     judge = CustomJudge("mock")
#     schema = Verdicts
#     answer = (
#         '{\n"verdicts": [\n{\n"verdict": "yes"\n},\n{\n    "verdict": "no",\n    "reason": "blah blah"\n},'
#         '\n{\n    "verdict": "yes",\n    "reason":null \n}\n]\n}'
#     )
#     res: Verdicts = judge.generate(answer, schema=schema)
