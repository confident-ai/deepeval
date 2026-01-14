import asyncio
import os

from typing import List
from openai import OpenAI

from tests.utils.trace_assertions import assert_trace_json
from tests.test_docs.test_deepeval.test_llm_evals.helpers import (
    find_span_by_name,
)
from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden
from deepeval.tracing import observe, update_current_span
from deepeval.tracing.trace_test_manager import trace_testing_manager
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


current_dir = os.path.dirname(os.path.abspath(__file__))
client = OpenAI()


def your_llm_app(input: str):
    @observe(type="retriever")
    def retriever(input: str):
        return ["Hardcoded text chunks from your vector database"]

    @observe(metrics=[AnswerRelevancyMetric()])
    def generator(input: str, retrieved_chunks: List[str]):
        res = (
            client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Use the provided context to answer the question.",
                    },
                    {
                        "role": "user",
                        "content": "\n\n".join(retrieved_chunks)
                        + "\n\nQuestion: "
                        + input,
                    },
                ],
            )
            .choices[0]
            .message.content
        )

        # Create test case at runtime
        update_current_span(
            test_case=LLMTestCase(input=input, actual_output=res)
        )

        return res

    return generator(input, retriever(input))


def _all_spans(trace_dict: dict):
    spans = []
    for key in (
        "llmSpans",
        "retrieverSpans",
        "toolSpans",
        "agentSpans",
        "baseSpans",
    ):
        spans.extend(trace_dict.get(key) or [])
    return spans


def _get_latest_trace_dict():
    """
    trace_testing_manager.test_dict is often populated synchronously,
    but we keep a fallback to the async wait to avoid flakes.
    """
    if trace_testing_manager.test_dict is not None:
        return trace_testing_manager.test_dict

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(trace_testing_manager.wait_for_test_dict())


###############################
# Setup LLM tracing & metrics #
###############################


@assert_trace_json(
    json_path=os.path.join(
        current_dir, "test_component_level_observed_generator.json"
    )
)
def test_observed_generator_span_emitted_shape():
    """
    Shape test: asserts a trace payload is emitted and contains the generator span structure.
    Only assert structure and types.
    """
    out = your_llm_app("How are you?")
    assert out == "MOCK_RESPONSE"


def test_observed_generator_span_emitted():
    """asserts a span is created for the @observed generator (name/type present)"""
    # Run the app
    user_input = "How are you?"
    out = your_llm_app(user_input)
    assert out == "MOCK_RESPONSE"

    trace_dict = trace_testing_manager.test_dict
    assert trace_dict is not None

    generator_span = find_span_by_name(trace_dict, "generator")
    assert generator_span is not None

    assert generator_span.get("name") == "generator"
    assert generator_span.get("input") == user_input
    assert generator_span.get("output") == "MOCK_RESPONSE"
    assert generator_span.get("status") == "SUCCESS"
    assert "metricsData" in generator_span


@assert_trace_json(
    json_path=os.path.join(
        current_dir, "test_component_level_update_current_span.json"
    )
)
def test_update_current_span_attaches_llm_test_case_to_generator_span_shape():
    """
    Shape test: asserts that calling update_current_span(test_case=...) results in the span
    including the expected fields in the emitted trace payload. Only assert structure and types.
    """
    out = your_llm_app("How are you?")
    assert out == "MOCK_RESPONSE"


def test_update_current_span_attaches_llm_test_case_to_generator_span():
    """asserts the generator span carries an attached LLMTestCase with correct input/output"""
    user_input = "How are you?"
    out = your_llm_app(user_input)
    assert out == "MOCK_RESPONSE"

    trace_dict = trace_testing_manager.test_dict
    assert (
        trace_dict is not None
    ), "Expected trace_testing_manager.test_dict to be populated."

    generator_span = find_span_by_name(trace_dict, "generator")
    assert (
        generator_span is not None
    ), "Expected a span named 'generator' to be emitted."

    # The runtime LLMTestCase is flattened onto the span in this implementation.
    assert (
        generator_span.get("input") == user_input
    ), f"Expected generator span input to be {user_input!r}, got {generator_span.get('input')!r}"
    assert (
        generator_span.get("output") == "MOCK_RESPONSE"
    ), f"Expected generator span output to be 'MOCK_RESPONSE', got {generator_span.get('output')!r}"

    metrics = generator_span.get("metricsData")
    # If metrics have a "name" or "metric" field in your payload, match it
    names = {m.get("name") for m in metrics}
    assert "Answer Relevancy" in names or "AnswerRelevancyMetric" in names


def test_metrics_configured_on_observe_execute_for_generator_span():
    """asserts AnswerRelevancyMetric produces metric data for the generator span"""
    out = your_llm_app("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = trace_testing_manager.test_dict
    assert trace_dict is not None, "Expected trace payload to be captured."

    generator_span = find_span_by_name(trace_dict, "generator")
    assert generator_span is not None, "Expected generator span to exist."

    metrics_data = generator_span.get("metricsData")
    assert (
        metrics_data is not None
    ), "Expected metricsData to be present on generator span."
    assert (
        len(metrics_data) >= 1
    ), "Expected at least one metric entry in metricsData."

    # Assert the Answer Relevancy metric is present
    metric_names = [m.get("name") for m in metrics_data]
    assert (
        "Answer Relevancy" in metric_names
    ), f"Expected 'Answer Relevancy' metric in metricsData. Got: {metric_names}"

    # Optional: assert stable config fields for that metric entry
    answer_rel = next(
        m for m in metrics_data if m.get("name") == "Answer Relevancy"
    )
    assert answer_rel.get("threshold") == 0.5
    assert answer_rel.get("strictMode") is False
    assert answer_rel.get("evaluationModel") == "gpt-4.1"
    # success might vary depending on how/when metrics are actually computed,
    # so only assert it exists as a boolean if you want:
    assert isinstance(answer_rel.get("success"), bool)


def test_metrics_not_applied_to_non_metric_components():
    """asserts retriever span exists but has no metric results"""
    out = your_llm_app("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = trace_testing_manager.test_dict
    assert trace_dict is not None, "Expected trace payload to be captured."

    print([s.get("name") for s in _all_spans(trace_dict)])
    spans = _all_spans(trace_dict)
    assert (
        spans
    ), f"Expected at least one span. Keys: {sorted(trace_dict.keys())}"

    # Only generator should have metricsData
    offenders = [
        s.get("name")
        for s in spans
        if s.get("name") != "generator"
        and (s.get("metricsData") or "metricsData" in s)
    ]
    assert (
        not offenders
    ), f"Expected no metricsData on non-generator spans; found on: {offenders}"

    # (Optional) if retriever span exists, assert it specifically
    retriever_span = next(
        (s for s in spans if s.get("name") == "retriever"), None
    )
    if retriever_span is not None:
        assert not retriever_span.get(
            "metricsData"
        ), "Expected retriever to have no metricsData"


####################
# Create a dataset #
####################


def test_goldens_single_turn_constructable():
    """
    Docs contract test:
    - You can construct single-turn Goldens with `Golden(input=...)`.
    - An EvaluationDataset can be created from those goldens.
    - evals_iterator() yields those goldens back in order with the expected inputs.
    """
    goldens = [
        Golden(input="What is your name?"),
        Golden(input="Choose a number between 1 to 100"),
    ]

    dataset = EvaluationDataset(goldens=goldens)

    seen = [g.input for g in dataset.evals_iterator()]
    assert seen == [g.input for g in goldens]


def test_conversational_golden_rejected_or_skipped_for_component_level():
    """
    Component-level eval tests expect a single-turn Golden with `.input`.
    ConversationalGolden may exist in the dataset, but it should not be runnable
    by the component-level harness.

    """
    dataset = EvaluationDataset(
        goldens=[
            ConversationalGolden(
                scenario="Andy Byron wants to purchase a VIP ticket to a Coldplay concert.",
                expected_outcome="Successful purchase of a ticket.",
                user_description="Andy Byron is the CEO of Astronomer.",
            )
        ]
    )

    goldens = list(dataset.evals_iterator())
    assert len(goldens) == 1
    g = goldens[0]

    assert isinstance(g, ConversationalGolden)
    assert isinstance(g, Golden) is False  # since it is not a single-turn


####################################################
# Run component-level evals using evals_iterator() #
####################################################


def test_evals_iterator_invokes_app_for_each_golden_and_emits_spans():
    """for N goldens, N generator spans (or N test cases) are recorded"""
    dataset = EvaluationDataset(
        goldens=[
            Golden(input="What is 1+1?"),
            Golden(input="What is the capital of France?"),
            Golden(input="Say 'hello'"),
        ]
    )

    generator_spans = []

    for golden in dataset.evals_iterator():
        # clear out previous trace capture so we're sure we read "this iteration"
        trace_testing_manager.test_dict = None

        out = your_llm_app(golden.input)
        assert out == "MOCK_RESPONSE"

        trace_dict = _get_latest_trace_dict()
        assert trace_dict is not None

        gen = find_span_by_name(trace_dict, "generator")
        assert gen is not None, (
            "Expected a 'generator' span per invocation. "
            f"Available spans: {[s.get('name') for k in ('llmSpans','retrieverSpans','toolSpans','agentSpans','baseSpans') for s in (trace_dict.get(k) or [])]}"
        )
        generator_spans.append(gen)

    assert len(generator_spans) == 3


def test_evals_iterator_optional_parameters_smoke():
    """
    passes the six documented options and ensures the loop runs:
    metrics, identifier, async_config, display_config, error_config, cache_config
    """
    dataset = EvaluationDataset(
        goldens=[Golden(input="Ping?"), Golden(input="Pong?")]
    )

    it = dataset.evals_iterator(
        metrics=[AnswerRelevancyMetric()],
        identifier="component-level-smoke",
        async_config=None,
        display_config=None,
        error_config=None,
        cache_config=None,
    )

    goldens = list(it)
    assert len(goldens) == 2
    assert [g.input for g in goldens] == ["Ping?", "Pong?"]
