from deepeval.dataset import Golden
from deepeval.tracing import observe, update_current_span_test_case
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.evaluate import evaluate
from deepeval.test_case import LLMTestCase

metric = AnswerRelevancyMetric()


@observe(metrics=[metric])
def inner_component():
    # Component can be anything from an LLM call, retrieval, agent, tool use, etc.
    update_current_span_test_case(
        LLMTestCase(input="Hi!", actual_output="Hi!", expected_output="Hi!")
    )
    return


@observe
def llm_app(input: str):
    inner_component()
    return


evaluate(observed_callback=llm_app, goldens=[Golden(input="Hi!")])
