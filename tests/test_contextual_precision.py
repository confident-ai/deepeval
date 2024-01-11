import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
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

four = """
Understanding comets and asteroids is crucial in studying the solar system's formation 
and evolution. Comets, which are remnants from the outer solar system, can provide 
insights into its icy and volatile components. Asteroids, primarily remnants of the 
early solar system's formation, offer clues about the materials that didn't form into 
planets, mostly located in the asteroid belt.
"""

five = """
The physical characteristics and orbital paths of comets and asteroids vary significantly. 
Comets often have highly elliptical orbits, taking them close to the Sun and then far into 
the outer solar system. Their icy composition leads to distinctive features like tails and 
comas. Asteroids, conversely, have more circular orbits and lack these visible features, 
being composed mostly of rock and metal.
"""


@pytest.mark.skip(reason="openai is expensive")
def test_contextual_precision():
    metric = ContextualPrecisionMetric(threshold=0.5)
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=answer,
        retrieval_context=[one, four, two, five, three],
    )
    assert_test(test_case, [metric])
