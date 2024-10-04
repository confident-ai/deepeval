import pytest
from langchain_openai import OpenAIEmbeddings

import deepeval
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
    GEval,
    SummarizationMetric,
    ToolCorrectnessMetric,
    ConversationRelevancyMetric,
    RoleAdherenceMetric,
    ConversationCompletenessMetric,
)
from deepeval.metrics.ragas import RagasMetric
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

strict_mode = False
verbose_mode = True


@pytest.mark.skip(reason="openai is expensive")
def test_everything():
    metric1 = AnswerRelevancyMetric(
        threshold=0.1,
        strict_mode=strict_mode,
        async_mode=False,
        verbose_mode=verbose_mode,
    )
    metric2 = FaithfulnessMetric(
        threshold=0.5, strict_mode=strict_mode, verbose_mode=verbose_mode
    )
    metric3 = ContextualPrecisionMetric(
        threshold=0.5, strict_mode=strict_mode, verbose_mode=verbose_mode
    )
    metric4 = ContextualRecallMetric(
        threshold=0.5, strict_mode=strict_mode, verbose_mode=verbose_mode
    )
    metric5 = ContextualRelevancyMetric(
        threshold=0.5, strict_mode=strict_mode, verbose_mode=verbose_mode
    )
    metric6 = BiasMetric(
        threshold=0.5, strict_mode=strict_mode, verbose_mode=verbose_mode
    )
    metric7 = ToxicityMetric(
        threshold=0.5, strict_mode=strict_mode, verbose_mode=verbose_mode
    )
    metric8 = HallucinationMetric(
        threshold=0.5, strict_mode=strict_mode, verbose_mode=verbose_mode
    )
    metric9 = SummarizationMetric(
        threshold=0.5, strict_mode=strict_mode, verbose_mode=verbose_mode
    )
    metric10 = GEval(
        name="Coherence",
        criteria="Coherence - determine if the actual output is coherent with the input, and does not contradict anything in the retrieval context.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        strict_mode=strict_mode,
        model="gpt-4-0125-preview",
        verbose_mode=verbose_mode,
    )

    metric11 = GEval(
        name="Relevancy",
        criteria="Relevancy - determine if the actual output is relevant with the input.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        strict_mode=strict_mode,
        model="gpt-4-0125-preview",
        verbose_mode=verbose_mode,
    )

    metric12 = ConversationRelevancyMetric()
    metric13 = ToolCorrectnessMetric()
    metric14 = ConversationCompletenessMetric()
    metric15 = RoleAdherenceMetric()

    test_case = LLMTestCase(
        input="What is this",
        actual_output="this is a latte",
        expected_output="this is a mocha",
        # retrieval_context=["I love coffee"],
        context=["I love coffee"],
        tools_called=["ok"],
        expected_tools=["ok", "ok"],
    )
    c_test_case = ConversationalTestCase(turns=[test_case, test_case])
    assert_test(
        c_test_case,
        [
            metric1,
            metric2,
            metric3,
            metric4,
            metric5,
            # metric6,
            # metric7,
            # metric8,
            # metric9,
            # metric10,
            # metric11,
            # metric12,
            # metric13,
            metric14,
            metric15,
        ],
        run_async=True,
    )


@pytest.mark.skip(reason="openadi is expensive")
def test_everything_2():
    metric1 = AnswerRelevancyMetric(threshold=0.5, strict_mode=strict_mode)
    metric2 = FaithfulnessMetric(threshold=0.5, strict_mode=strict_mode)
    metric3 = ContextualPrecisionMetric(threshold=0.5, strict_mode=strict_mode)
    metric4 = ContextualRecallMetric(threshold=0.5, strict_mode=strict_mode)
    metric5 = ContextualRelevancyMetric(threshold=0.1, strict_mode=strict_mode)
    metric6 = BiasMetric(threshold=0.2, strict_mode=strict_mode)
    metric7 = ToxicityMetric(threshold=0.5, strict_mode=strict_mode)
    metric8 = HallucinationMetric(threshold=0.5, strict_mode=strict_mode)
    metric9 = SummarizationMetric(threshold=0.5, strict_mode=strict_mode, n=2)
    metric10 = (
        GEval(
            name="Coherence",
            criteria="Coherence - determine if the actual output is coherent with the input.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            strict_mode=True,
        ),
    )
    metric11 = RagasMetric(
        threshold=0.5, model="gpt-3.5-turbo", embeddings=OpenAIEmbeddings()
    )
    metric12 = ToolCorrectnessMetric()

    test_case = LLMTestCase(
        input="What is this again?",
        actual_output="this is a latte",
        expected_output="this is a mocha",
        # retrieval_context=["I love coffee"],
        context=["I love coffee"],
        expected_tools=["mixer", "creamer", "dripper"],
        tools_called=["mixer", "creamer", "mixer"],
    )
    c_test_case = ConversationalTestCase(
        name="testing_", turns=[test_case, test_case]
    )
    assert_test(
        test_case,
        [
            metric1,
            # metric2,
            # metric3,
            # metric4,
            # metric5,
            # metric6,
            # metric7,
            # metric8,
            # metric9,
            # metric10,
            # metric11,
            metric12,
        ],
        # run_async=False,
    )


@deepeval.log_hyperparameters(
    model="gpt-4", prompt_template="another template!"
)
def hyperparameters():
    return {"chunk_size": 600, "temperature": 1}
