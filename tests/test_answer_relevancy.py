"""Tests for answer relevancy
"""

import pytest
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import assert_test

question = "What are the primary benefits of meditation?"
answer = """
Meditation offers a rich tapestry of benefits that touch upon various aspects of well-being. On a mental level, 
it greatly reduces stress and anxiety, fostering enhanced emotional health. This translates to better emotional 
regulation and a heightened sense of overall well-being. Interestingly, the practice of meditation has been around 
for centuries, evolving through various cultures and traditions, which underscores its timeless relevance.

Physically, it contributes to lowering blood pressure and alleviating chronic pain, which is pivotal for long-term health. 
Improved sleep quality is another significant benefit, aiding in overall physical restoration. Cognitively, meditation is a 
boon for enhancing attention span, improving memory, and slowing down age-related cognitive decline. Amidst these benefits, 
meditation's role in cultural and historical contexts is a fascinating side note, though not directly related to its health benefits.

Such a comprehensive set of advantages makes meditation a valuable practice for individuals seeking holistic improvement i
n both mental and physical health, transcending its historical and cultural origins.
"""

one = """
Meditation is an ancient practice, rooted in various cultural traditions, where individuals 
engage in mental exercises like mindfulness or concentration to promote mental clarity, emotional 
calmness, and physical relaxation. This practice can range from techniques focusing on breath, visual 
imagery, to movement-based forms like yoga. The goal is to bring about a sense of peace and self-awareness, 
enabling individuals to deal with everyday stress more effectively.
"""

two = """
One of the key benefits of meditation is its impact on mental health. It's widely used as a tool to 
reduce stress and anxiety. Meditation helps in managing emotions, leading to enhanced emotional health. 
It can improve symptoms of anxiety and depression, fostering a general sense of well-being. Regular practice 
is known to increase self-awareness, helping individuals understand their thoughts and emotions more clearly 
and reduce negative reactions to challenging situations.
"""

three = """
Meditation has shown positive effects on various aspects of physical health. It can lower blood pressure, 
reduce chronic pain, and improve sleep. From a cognitive perspective, meditation can sharpen the mind, increase 
attention span, and improve memory. It's particularly beneficial in slowing down age-related cognitive decline and 
enhancing brain functions related to concentration and attention.
"""


# Inherit BaseMetric
class FakeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        self.score = 1
        self.success = self.score >= self.threshold
        self.reason = "This metric looking good!"
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        self.score = 1
        self.success = self.score >= self.threshold
        self.reason = "This async metric looking good!"
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Fake"


@pytest.mark.skip(reason="openai is very expensive")
def test_answer_relevancy():
    metric = FakeMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What is your name",
        actual_output="Idk",
        retrieval_context=[one, two, three],
    )
    assert_test(test_case, [metric])
