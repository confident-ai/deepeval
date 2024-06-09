import pytest
from langchain_openai import OpenAIEmbeddings
import asyncio
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

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
    BaseMetric
)
from deepeval.dataset import EvaluationDataset
from deepeval.metrics.ragas import RagasMetric
from deepeval import assert_test, evaluate

#########################################################
# test cases
##########################################################

test_case_1 = LLMTestCase(
        input="What is this again?",
        actual_output="This is a latte.",
        expected_output="This is a mocha.",
        retrieval_context=["I love coffee."],
        context=["I love coffee."],
    )

test_case_2 = LLMTestCase(
        input="What is this again?",
        actual_output="This is a latte.",
        expected_output="This is a latte.",
        retrieval_context=["I love coffee."],
        context=["I love coffee."],
    )

test_case_3 = LLMTestCase(
        input="Do you have cold brew?",
        actual_output="Cold brew has numerous health benefits.",
        expected_output="No, we only have latte and americano",
        retrieval_context=["I love coffee."],
        context=["Our drinks include latte and americano"],
    )

test_case_4 = LLMTestCase(
        input="Can I get an americano with almond milk?",
        actual_output="We don't have almond milk.",
        expected_output="Yes, we can make an americano with almond milk.",
        retrieval_context=["We have soy and oat milk."],
        context=["We offer various milk options, including almond milk."]
    )

test_case_5 = LLMTestCase(
        input="Is the espresso strong?",
        actual_output="Our espresso is mild.",
        expected_output="Yes, our espresso is quite strong.",
        retrieval_context=["Some customers find our coffee mild."],
        context=["Our espresso is known for its strong flavor."]
    )

test_case_6 = LLMTestCase(
        input="What desserts do you have?",
        actual_output="We have cakes and cookies.",
        expected_output="We have cakes, cookies, and brownies.",
        retrieval_context=["Our cafe offers cookies and brownies."],
        context=["Our dessert options include cakes, cookies, and brownies."]
    )

test_case_7 = LLMTestCase(
        input="Do you serve breakfast all day?",
        actual_output="Breakfast is only served until 11 AM.",
        expected_output="Yes, we serve breakfast all day.",
        retrieval_context=["Breakfast times are usually until noon."],
        context=["Breakfast is available all day at our cafe."]
    )

test_case_8 = LLMTestCase(
        input="Do you have any vegan options?",
        actual_output="We have vegan salads.",
        expected_output="We offer vegan salads and smoothies.",
        retrieval_context=["We recently started offering vegan dishes."],
        context=["We have vegan salads and smoothies on our menu."]
    )

test_case_9 = LLMTestCase(
        input="Is there parking nearby?",
        actual_output="There is no parking available.",
        expected_output="Yes, there is a parking lot right behind the cafe.",
        retrieval_context=["Street parking can be hard to find."],
        context=["Parking is available behind the cafe."]
    )

test_cases = [
    test_case_1,
    test_case_2,
    test_case_3,
    # test_case_4,
    # test_case_5,
    # test_case_6,
    # test_case_7,
    # test_case_8,
    # test_case_9
]

##########################################################
# a_measure 
##########################################################

async def single_async_call(
        metric: BaseMetric,
        test_case: LLMTestCase):
    await metric.a_measure(test_case)
    print("metric: " + metric.__name__)
    print("score: " + str(metric.score))
    print("reason: " + metric.reason)

async def test_async():
    # metrics
    strict_mode = False
    metric1 = AnswerRelevancyMetric(threshold=0.1, strict_mode=strict_mode)
    metric2 = FaithfulnessMetric(threshold=0.5, strict_mode=strict_mode)
    metric3 = ContextualPrecisionMetric(threshold=0.5, strict_mode=strict_mode)
    metric4 = ContextualRecallMetric(threshold=0.5, strict_mode=strict_mode)
    metric5 = ContextualRelevancyMetric(threshold=0.5, strict_mode=strict_mode)
    metric6 = BiasMetric(threshold=0.5, strict_mode=strict_mode)
    metric7 = ToxicityMetric(threshold=0.5, strict_mode=strict_mode)
    metric8 = HallucinationMetric(threshold=0.5, strict_mode=strict_mode)
    metric9 = SummarizationMetric(threshold=0.5, strict_mode=strict_mode)
    metric10 = GEval(
        name="Coherence",
        criteria="Coherence - determine if the actual output is coherent with the input, and does not contradict anything in the retrieval context.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
    )

    # Prepare the list of metrics based on even and odd indexed test cases
    tasks = []
    for i, test_case in enumerate(test_cases):
        if i % 2 == 0:  # Even index
            tasks.append(single_async_call(metric1, test_case))
        else:  # Odd index
            tasks.append(single_async_call(metric2, test_case))

    # Execute all tasks asynchronously
    await asyncio.gather(*tasks)

#asyncio.run(test_async())

# ##########################################################
# # measure in sync mode
# ##########################################################

def single_sync_call(
        metric: BaseMetric,
        test_case: LLMTestCase):
    metric.measure(test_case)
    print("metric: " + metric.__name__)
    print("score: " + str(metric.score))
    print("reason: " + metric.reason)

def test_sync():
    # metrics
    strict_mode = False
    metric1 = AnswerRelevancyMetric(threshold=0.1, strict_mode=strict_mode, async_mode=False)
    metric2 = FaithfulnessMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
    metric3 = ContextualPrecisionMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
    metric4 = ContextualRecallMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
    metric5 = ContextualRelevancyMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
    metric6 = BiasMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
    metric7 = ToxicityMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
    metric8 = HallucinationMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
    metric9 = SummarizationMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
    metric10 = GEval(
        name="Coherence",
        criteria="Coherence - determine if the actual output is coherent with the input, and does not contradict anything in the retrieval context.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        async_mode=False
    )

    # Prepare the list of metrics based on even and odd indexed test cases
    tasks = []
    for i, test_case in enumerate(test_cases):
        if i % 2 == 0:  # Even index
            tasks.append(single_sync_call(metric1, test_case))
        else:  # Odd index
            tasks.append(single_sync_call(metric2, test_case))

#test_sync()

# ##########################################################
# # measure in async mode
# ##########################################################

def single_sync_in_async_call(
        metric: BaseMetric,
        test_case: LLMTestCase):
    metric.measure(test_case)
    print("metric: " + metric.__name__)
    print("score: " + str(metric.score))
    print("reason: " + metric.reason)

def test_sync_in_async():
    # metrics
    strict_mode = False
    metrics = [
        AnswerRelevancyMetric(threshold=0.1, strict_mode=strict_mode, async_mode=True),
        FaithfulnessMetric(threshold=0.5, strict_mode=strict_mode),
        ContextualPrecisionMetric(threshold=0.5, strict_mode=strict_mode),
        ContextualRecallMetric(threshold=0.5, strict_mode=strict_mode),
        ContextualRelevancyMetric(threshold=0.5, strict_mode=strict_mode),
        BiasMetric(threshold=0.5, strict_mode=strict_mode),
        ToxicityMetric(threshold=0.5, strict_mode=strict_mode),
        HallucinationMetric(threshold=0.5, strict_mode=strict_mode),
        SummarizationMetric(threshold=0.5, strict_mode=strict_mode),
        GEval(
            name="Coherence",
            criteria="Coherence - determine if the actual output is coherent with the input, and does not contradict anything in the retrieval context.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            async_mode=False  # Assuming it allows synchronous operations
        )
    ]

    tasks = []
    # Test every metric on every test case
    for metric in metrics:
        tasks.append(single_sync_in_async_call(metric, test_case_1))

#test_sync_in_async()

# ##########################################################
# # evaluate
# ##########################################################

strict_mode = False
metric1 = AnswerRelevancyMetric(threshold=0.1, strict_mode=strict_mode, async_mode=False)
metric2 = FaithfulnessMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
metric3 = ContextualPrecisionMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
metric4 = ContextualRecallMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
metric5 = ContextualRelevancyMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
metric6 = BiasMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
metric7 = ToxicityMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
metric8 = HallucinationMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
metric9 = SummarizationMetric(threshold=0.5, strict_mode=strict_mode, async_mode=False)
metric10 = GEval(
    name="Coherence",
    criteria="Coherence - determine if the actual output is coherent with the input, and does not contradict anything in the retrieval context.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    async_mode=False
)
    
dataset = EvaluationDataset(test_cases=test_cases)
#dataset.evaluate([metric1, metric2])
#evaluate(dataset, [metric1, metric2])
#evaluate(dataset, [metric1, metric2], run_async=False, show_indicator=True)

##########################################################
# deep eval test run
##########################################################
strict_mode = False

#@pytest.mark.skip(reason="openai is expensive")
def test_everything():
    metric1 = AnswerRelevancyMetric(
        threshold=0.1,
        strict_mode=strict_mode,
        async_mode=False,
        model="gpt-4-0125-preview",
    )
    metric2 = FaithfulnessMetric(threshold=0.5, strict_mode=strict_mode)
    metric3 = ContextualPrecisionMetric(threshold=0.5, strict_mode=strict_mode)
    metric4 = ContextualRecallMetric(threshold=0.5, strict_mode=strict_mode)
    metric5 = ContextualRelevancyMetric(threshold=0.5, strict_mode=strict_mode)
    metric6 = BiasMetric(threshold=0.5, strict_mode=strict_mode)
    metric7 = ToxicityMetric(threshold=0.5, strict_mode=strict_mode)
    metric8 = HallucinationMetric(threshold=0.5, strict_mode=strict_mode)
    metric9 = SummarizationMetric(threshold=0.5, strict_mode=strict_mode)
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
    )

    test_case = LLMTestCase(
        input="What is this",
        actual_output="this is a latte",
        expected_output="this is a mocha",
        retrieval_context=["I love coffee"],
        context=["I love coffee"],
    )
    c_test_case = ConversationalTestCase(messages=[test_case, test_case])
    assert_test(
        c_test_case,
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
            metric10,
        ],
        # run_async=False,
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

    test_case = LLMTestCase(
        input="What is this again?",
        actual_output="this is a latte",
        expected_output="this is a mocha",
        retrieval_context=["I love coffee"],
        context=["I love coffee"],
    )
    assert_test(
        test_case,
        [
            metric1,
            metric2,
            metric3,
            metric4,
            # metric5,
            metric6,
            # metric7,
            # metric8,
            # metric9,
            # metric10,
            # metric11,
        ],
        run_async=False,
    )


@deepeval.log_hyperparameters(
    model="gpt-4", prompt_template="another template!"
)
def hyperparameters():
    return {"chunk_size": 600, "temperature": 1}
