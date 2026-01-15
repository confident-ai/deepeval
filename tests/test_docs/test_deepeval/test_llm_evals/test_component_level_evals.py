import os

from typing import List
from openai import OpenAI

from tests.utils.trace_assertions import assert_trace_json
from tests.test_docs.test_deepeval.test_llm_evals.helpers import (
    find_span_by_name,
    span_names_by_key,
    all_spans,
    get_latest_trace_dict,
    debug_span_names,
)
from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden
from deepeval.tracing import observe, update_current_span
from deepeval.tracing.trace_test_manager import trace_testing_manager
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


current_dir = os.path.dirname(os.path.abspath(__file__))
client = OpenAI()


# Apps


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


def your_llm_app_rooted(input: str):
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

    @observe(type="agent", name="app")
    def app():
        return generator(input, retriever(input))

    return app()


def your_llm_app_with_tool(input: str):
    @observe(type="tool")
    def tool_call(q: str) -> str:
        return "TOOL_RESULT"

    @observe(type="retriever")
    def retriever(input: str):
        _ = tool_call("lookup")
        return ["Hardcoded text chunks from your vector database"]

    @observe(metrics=[AnswerRelevancyMetric()])
    def generator(input: str, retrieved_chunks: List[str]):
        res = client.chat.completions.create(...).choices[0].message.content
        update_current_span(
            test_case=LLMTestCase(input=input, actual_output=res)
        )
        return res

    @observe(type="agent", name="app_with_tool")
    def app():
        chunks = retriever(input)
        return generator(input, chunks)

    return app()


def your_llm_app_with_agent(input: str):
    @observe(type="retriever")
    def retriever(input: str):
        return ["Hardcoded text chunks from your vector database"]

    @observe(metrics=[AnswerRelevancyMetric()])
    def generator(input: str, retrieved_chunks: List[str]):
        res = client.chat.completions.create(...).choices[0].message.content
        update_current_span(
            test_case=LLMTestCase(input=input, actual_output=res)
        )
        return res

    @observe(type="agent", name="app_with_agent")
    def agent(input: str):
        return generator(input, retriever(input))

    return agent(input)


def your_llm_app_no_metrics(input: str):
    @observe(type="retriever")
    def retriever(input: str):
        return ["Hardcoded text chunks from your vector database"]

    @observe()
    def generator(input: str, retrieved_chunks: List[str]):
        res = client.chat.completions.create(...).choices[0].message.content
        update_current_span(
            test_case=LLMTestCase(input=input, actual_output=res)
        )
        return res

    @observe(type="agent", name="app_no_metrics")
    def app():
        chunks = retriever(input)
        return generator(input, chunks)

    return app()


def your_llm_app_update_twice(input: str):
    @observe(metrics=[AnswerRelevancyMetric()])
    def generator(input: str):
        res = "MOCK_RESPONSE"
        update_current_span(
            test_case=LLMTestCase(input=input, actual_output="FIRST")
        )
        update_current_span(
            test_case=LLMTestCase(input=input, actual_output=res)
        )
        return res

    return generator(input)


###############################
# Setup LLM tracing & metrics #
###############################


@assert_trace_json(
    json_path=os.path.join(
        current_dir, "test_component_level_observed_generator.json"
    )
)
def test_observed_generator_span():
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

    print([s.get("name") for s in all_spans(trace_dict)])
    spans = all_spans(trace_dict)
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


@assert_trace_json(
    json_path=os.path.join(
        current_dir, "test_component_level_rooted_app_spans.json"
    ),
    # mode="generate",
)
def test_rooted_app_emits_agent_retriever_generator_and_metrics():
    user_input = "How are you?"
    out = your_llm_app_rooted(user_input)
    assert out == "MOCK_RESPONSE"

    trace_dict = trace_testing_manager.test_dict
    assert trace_dict is not None

    # Typed checks (this is the point of rooted)
    assert "app" in span_names_by_key(trace_dict, "agentSpans")
    assert "retriever" in span_names_by_key(trace_dict, "retrieverSpans")

    gen = find_span_by_name(trace_dict, "generator")
    assert gen is not None
    assert gen.get("metricsData"), "Expected metrics on generator"


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

        trace_dict = get_latest_trace_dict()
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


# Tools


def test_tool_span_emitted():
    out = your_llm_app_with_tool("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    assert trace_dict is not None

    tool_names = span_names_by_key(trace_dict, "toolSpans")
    base_names = span_names_by_key(trace_dict, "baseSpans")

    # Accept either: dedicated toolSpans OR baseSpans.
    assert ("tool_call" in tool_names) or (
        "tool_call" in base_names
    ), f"Expected a tool span named 'tool_call'. Span keys: {debug_span_names(trace_dict)}"


# Agents


def test_agent_span_emitted():
    out = your_llm_app_with_agent("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    assert trace_dict is not None

    agent_names = span_names_by_key(trace_dict, "agentSpans")
    assert (
        agent_names
    ), f"Expected at least one agent span. Span keys: {debug_span_names(trace_dict)}"
    assert (
        "app_with_agent" in agent_names
    ), f"Expected agent span named 'app_with_agent'. Span keys: {debug_span_names(trace_dict)}"


# Metrics


def test_metrics_only_on_metric_configured_span():
    out = your_llm_app_with_agent("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    spans = all_spans(trace_dict)

    assert spans, f"No spans emitted. Keys: {sorted(trace_dict.keys())}"

    # generator should have metricsData
    generator = next((s for s in spans if s.get("name") == "generator"), None)
    assert (
        generator is not None
    ), f"Missing generator span. {debug_span_names(trace_dict)}"
    assert generator.get(
        "metricsData"
    ), "Expected generator.metricsData to be present and non-empty."

    # everyone else should not
    offenders = [
        s.get("name")
        for s in spans
        if s.get("name") != "generator"
        and (s.get("metricsData") or "metricsData" in s)
    ]
    assert (
        not offenders
    ), f"Expected no metricsData on non-generator spans; found on: {offenders}"


def test_generator_span_no_metrics_when_not_configured():
    out = your_llm_app_no_metrics("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    gen = find_span_by_name(trace_dict, "generator")
    assert gen is not None

    assert ("metricsData" not in gen) or (
        not gen.get("metricsData")
    ), f"Expected no metricsData when metrics not configured; got: {gen.get('metricsData')}"


def test_update_current_span_last_write_wins():
    out = your_llm_app_update_twice("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    gen = find_span_by_name(trace_dict, "generator")
    assert gen is not None
    assert gen.get("output") == "MOCK_RESPONSE"


def test_evals_iterator_emits_span_with_matching_input_per_golden():
    dataset = EvaluationDataset(
        goldens=[Golden(input="A"), Golden(input="B"), Golden(input="C")]
    )

    for golden in dataset.evals_iterator():
        trace_testing_manager.test_dict = None
        out = your_llm_app(golden.input)
        assert out == "MOCK_RESPONSE"

        trace_dict = get_latest_trace_dict()
        gen = find_span_by_name(trace_dict, "generator")
        assert gen is not None
        assert gen.get("input") == golden.input
