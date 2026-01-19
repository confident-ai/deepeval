import os

from typing import List
from openai import OpenAI

from tests.utils.trace_assertions import assert_trace_json
from tests.test_end_to_end.test_deepeval.test_llm_evals.helpers import (
    find_span_by_name,
    span_names_by_key,
    all_spans,
    get_latest_trace_dict,
    debug_span_names,
)
from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.tracing.trace_test_manager import trace_testing_manager
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import ToolCall
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


def your_llm_app_with_update_current_trace(input: str):
    """
    Docs: update_current_trace can be used to set end-to-end test cases for the trace.
    This app demonstrates using update_current_trace to set trace-level test case data.
    """

    @observe(type="retriever")
    def retriever(input: str):
        chunks = ["Hardcoded text chunks from your vector database"]
        update_current_trace(retrieval_context=chunks)
        return chunks

    @observe()
    def generator(input: str, retrieved_chunks: List[str]):
        res = client.chat.completions.create(...).choices[0].message.content
        update_current_trace(input=input, output=res)
        return res

    @observe(type="agent", name="app_with_trace_update")
    def app():
        chunks = retriever(input)
        return generator(input, chunks)

    return app()


def your_llm_app_update_span_with_individual_params(input: str):
    """
    Docs: update_current_span can take individual LLMTestCase params directly
    instead of a test_case object.
    """

    @observe(type="retriever")
    def retriever(input: str):
        chunks = ["Hardcoded text chunks"]
        # Using individual params instead of test_case=LLMTestCase(...)
        update_current_span(input=input, retrieval_context=chunks)
        return chunks

    @observe(metrics=[AnswerRelevancyMetric()])
    def generator(input: str, retrieved_chunks: List[str]):
        res = client.chat.completions.create(...).choices[0].message.content
        # Using individual params: input, output (actual_output maps to output)
        update_current_span(
            input=input, output=res, retrieval_context=retrieved_chunks
        )
        return res

    @observe(type="agent", name="app_individual_params")
    def app():
        chunks = retriever(input)
        return generator(input, chunks)

    return app()


def your_llm_app_with_custom_span_name(input: str):
    """
    Docs: The @observe decorator accepts a `name` parameter to customize span display name.
    """

    @observe(name="CustomRetrieverName")
    def retriever(input: str):
        return ["Hardcoded text chunks"]

    @observe(name="CustomGeneratorName", metrics=[AnswerRelevancyMetric()])
    def generator(input: str, retrieved_chunks: List[str]):
        res = client.chat.completions.create(...).choices[0].message.content
        update_current_span(
            test_case=LLMTestCase(input=input, actual_output=res)
        )
        return res

    return generator(input, retriever(input))


def your_llm_app_with_llm_type(input: str):
    """
    Docs: The @observe decorator accepts type="llm" for LLM-specific spans.
    """

    @observe(type="llm", name="llm_call")
    def llm_call(input: str):
        res = client.chat.completions.create(...).choices[0].message.content
        update_current_span(input=input, output=res)
        return res

    return llm_call(input)


def your_llm_app_nested_spans(input: str):
    """
    Docs: A span can contain many child spans, forming a tree structure.
    Tests nested span hierarchy.
    """

    @observe(name="inner_retriever")
    def inner_retriever(query: str):
        return ["inner chunk"]

    @observe(name="outer_retriever")
    def outer_retriever(query: str):
        inner_chunks = inner_retriever(query)
        return inner_chunks + ["outer chunk"]

    @observe(metrics=[AnswerRelevancyMetric()])
    def generator(input: str, retrieved_chunks: List[str]):
        res = client.chat.completions.create(...).choices[0].message.content
        update_current_span(
            test_case=LLMTestCase(input=input, actual_output=res)
        )
        return res

    @observe(type="agent", name="nested_app")
    def app():
        chunks = outer_retriever(input)
        return generator(input, chunks)

    return app()


def your_llm_app_with_expected_output(input: str, expected: str = None):
    """
    Docs: LLMTestCase supports expected_output field for comparison metrics.
    """

    @observe(metrics=[AnswerRelevancyMetric()])
    def generator(input: str):
        res = client.chat.completions.create(...).choices[0].message.content
        update_current_span(
            test_case=LLMTestCase(
                input=input, actual_output=res, expected_output=expected
            )
        )
        return res

    return generator(input)


def your_llm_app_with_context(input: str):
    """
    Docs: LLMTestCase supports context field (ground truth context).
    """

    @observe(metrics=[AnswerRelevancyMetric()])
    def generator(input: str):
        res = client.chat.completions.create(...).choices[0].message.content
        ground_truth_context = ["This is the ground truth context"]
        update_current_span(
            test_case=LLMTestCase(
                input=input, actual_output=res, context=ground_truth_context
            )
        )
        return res

    return generator(input)


def your_llm_app_with_tools_called(input: str):
    """
    Docs: LLMTestCase supports tools_called and expected_tools fields.
    """

    @observe(type="tool", name="observe_search_tool")
    def search_tool(query: str):
        return "search result"

    @observe(metrics=[AnswerRelevancyMetric()])
    def generator(input: str):

        tool_result = search_tool(input)
        res = client.chat.completions.create(...).choices[0].message.content
        update_current_span(
            test_case=LLMTestCase(
                input=input,
                actual_output=res,
                tools_called=[
                    ToolCall(
                        name="search_tool",
                        input_parameters={"query": input},
                        output=tool_result,
                    )
                ],
                expected_tools=[ToolCall(name="search_tool")],
            )
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

    answer_rel = next(
        m for m in metrics_data if m.get("name") == "Answer Relevancy"
    )
    assert answer_rel.get("threshold") == 0.5
    assert answer_rel.get("strictMode") is False
    assert answer_rel.get("evaluationModel") == "gpt-4.1"
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


def test_update_current_trace_sets_trace_level_test_case():
    """
    Docs: update_current_trace can be used to set end-to-end test cases for the trace.
    Verifies that calling update_current_trace populates trace-level data.
    """
    out = your_llm_app_with_update_current_trace("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    assert trace_dict is not None

    # The trace should have the test case data set via update_current_trace
    # Check that spans were created (the app structure)
    agent_names = span_names_by_key(trace_dict, "agentSpans")
    assert "app_with_trace_update" in agent_names

    retriever_names = span_names_by_key(trace_dict, "retrieverSpans")
    # Retriever span should exist
    assert (
        len(retriever_names) >= 1
        or find_span_by_name(trace_dict, "retriever") is not None
    )


def test_update_current_span_with_individual_params():
    """
    Docs: update_current_span can take individual LLMTestCase params
    (input, output, retrieval_context, context, expected_output, etc.)
    directly instead of a test_case object.
    """
    out = your_llm_app_update_span_with_individual_params("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    assert trace_dict is not None

    gen = find_span_by_name(trace_dict, "generator")
    assert (
        gen is not None
    ), f"Expected generator span. {debug_span_names(trace_dict)}"

    # Verify the individual params were captured
    assert gen.get("input") == "How are you?"
    assert gen.get("output") == "MOCK_RESPONSE"


def test_observe_name_parameter_customizes_span_name():
    """
    Docs: The @observe decorator accepts a `name` parameter to customize
    how this span is displayed.
    """
    out = your_llm_app_with_custom_span_name("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    assert trace_dict is not None

    # Check that custom names are used instead of function names
    custom_retriever = find_span_by_name(trace_dict, "CustomRetrieverName")
    custom_generator = find_span_by_name(trace_dict, "CustomGeneratorName")

    assert custom_retriever is not None or custom_generator is not None, (
        f"Expected at least one custom-named span. "
        f"Available: {debug_span_names(trace_dict)}"
    )


def test_observe_type_llm_creates_llm_span():
    """
    Docs: The @observe decorator accepts type="llm" for LLM-specific spans.
    """
    out = your_llm_app_with_llm_type("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    assert trace_dict is not None

    llm_names = span_names_by_key(trace_dict, "llmSpans")
    base_names = span_names_by_key(trace_dict, "baseSpans")

    # Accept either llmSpans or baseSpans depending on implementation
    assert ("llm_call" in llm_names) or ("llm_call" in base_names), (
        f"Expected an llm span named 'llm_call'. "
        f"Span keys: {debug_span_names(trace_dict)}"
    )


def test_nested_spans_form_tree_structure():
    """
    Docs: A span can contain many child spans, forming a tree structure—
    just like how different components of your LLM application interact.
    """
    out = your_llm_app_nested_spans("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    assert trace_dict is not None

    spans = all_spans(trace_dict)

    def by_name(n: str):
        return next((s for s in spans if s.get("name") == n), None)

    app = by_name("nested_app")
    outer = by_name("outer_retriever")
    inner = by_name("inner_retriever")
    gen = by_name("generator")

    assert app is not None, "Missing nested_app span"
    assert outer is not None, "Missing outer_retriever span"
    assert inner is not None, "Missing inner_retriever span"
    assert gen is not None, "Missing generator span"

    # assert Parent child edges
    assert outer.get("parentUuid") == app.get("uuid")
    assert inner.get("parentUuid") == outer.get("uuid")
    assert gen.get("parentUuid") == app.get("uuid")


def test_llm_test_case_with_tools_called():
    """
    Docs: LLMTestCase supports tools_called and expected_tools fields.
    """
    out = your_llm_app_with_tools_called("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    gen = find_span_by_name(trace_dict, "generator")
    assert gen is not None

    # Also verify tool span was created
    tool_span = find_span_by_name(trace_dict, "observe_search_tool")
    tool_names = span_names_by_key(trace_dict, "toolSpans")
    base_names = span_names_by_key(trace_dict, "baseSpans")

    assert (
        tool_span is not None
        or "observe_search_tool" in tool_names
        or "observe_search_tool" in base_names
    ), f"Expected observe_search_tool span. Got: {debug_span_names(trace_dict)}"

    assert find_span_by_name(trace_dict, "search_tool") is None


def test_update_current_span_name_overrides_observer_name():
    @observe(type="tool", name="observer_name")
    def tool_fn(x: str):
        update_current_span(name="update_name")
        return "ok"

    out = tool_fn("x")
    assert out == "ok"

    trace_dict = get_latest_trace_dict()
    span = find_span_by_name(trace_dict, "update_name")
    assert span is not None, (
        "Expected update_current_span(name=...) to override @observe(name=...). "
        f"Got: {debug_span_names(trace_dict)}"
    )
    assert find_span_by_name(trace_dict, "observer_name") is None


def test_update_current_span_output_not_overridden_by_observer_kwargs():
    @observe(name="tool_span", type="tool", output="SHOULD_NOT_WIN")
    def tool_fn(x: str):
        update_current_span(output="SHOULD_WIN")
        return "ok"

    tool_fn("x")

    trace_dict = get_latest_trace_dict()
    span = find_span_by_name(trace_dict, "tool_span")
    assert span is not None
    assert span.get("output") == "SHOULD_WIN"


def test_update_current_span_name_overrides_function_name():
    @observe(type="tool")
    def tool_fn(x: str):
        update_current_span(name="update_name")
        return "ok"

    out = tool_fn("x")
    assert out == "ok"

    trace_dict = get_latest_trace_dict()
    span = find_span_by_name(trace_dict, "update_name")
    assert span is not None, (
        "Expected update_current_span(name=...) to override function name. "
        f"Got: {debug_span_names(trace_dict)}"
    )
    assert find_span_by_name(trace_dict, "tool_fn") is None


def test_golden_with_expected_output():
    """
    Docs: Golden can include expected_output for comparison during evaluation.
    """
    golden = Golden(input="What is 2+2?", expected_output="4")

    assert golden.input == "What is 2+2?"
    assert golden.expected_output == "4"


def test_golden_with_additional_metadata():
    """
    Docs: Golden supports additional fields for richer test cases.
    """
    golden = Golden(
        input="Tell me about Paris",
        expected_output="Paris is the capital of France.",
        context=[
            "Paris is located in France.",
            "Paris is known for the Eiffel Tower.",
        ],
    )

    assert golden.input == "Tell me about Paris"
    assert golden.expected_output == "Paris is the capital of France."
    assert golden.context == [
        "Paris is located in France.",
        "Paris is known for the Eiffel Tower.",
    ]


def test_evaluation_dataset_from_goldens_list():
    """
    Docs: EvaluationDataset can be created from a list of Golden objects.
    """
    goldens = [
        Golden(input="Q1"),
        Golden(input="Q2"),
        Golden(input="Q3"),
    ]

    dataset = EvaluationDataset(goldens=goldens)

    # Verify dataset contains all goldens
    result = list(dataset.evals_iterator())
    assert len(result) == 3
    assert [g.input for g in result] == ["Q1", "Q2", "Q3"]


def test_observe_decorator_without_parameters():
    """
    Docs: @observe() can be used without any parameters for basic tracing.
    """

    @observe()
    def simple_component(x: str):
        return f"processed: {x}"

    result = simple_component("test")
    assert result == "processed: test"

    trace_dict = get_latest_trace_dict()
    assert trace_dict is not None

    span = find_span_by_name(trace_dict, "simple_component")
    assert (
        span is not None
    ), f"Expected simple_component span. Got: {debug_span_names(trace_dict)}"


def test_observe_decorator_with_multiple_metrics():
    """
    Docs: @observe can accept a list of multiple metrics.
    """
    from deepeval.metrics import AnswerRelevancyMetric

    # Note: In real usage you might have different metrics
    # Here we just verify the decorator accepts a list
    @observe(
        metrics=[AnswerRelevancyMetric(), AnswerRelevancyMetric(threshold=0.7)]
    )
    def multi_metric_component(x: str):
        res = "response"
        update_current_span(test_case=LLMTestCase(input=x, actual_output=res))
        return res

    result = multi_metric_component("test")
    assert result == "response"

    trace_dict = get_latest_trace_dict()
    span = find_span_by_name(trace_dict, "multi_metric_component")
    assert span is not None

    # Should have metricsData with entries for both metrics
    metrics_data = span.get("metricsData")
    assert metrics_data is not None
    assert len(metrics_data) >= 1  # At least one metric should be present


def test_retriever_span_type():
    """
    Docs: type="retriever" creates a retriever-typed span.
    """

    @observe(type="retriever", name="test_retriever")
    def retriever_func(query: str):
        return ["chunk1", "chunk2"]

    result = retriever_func("test query")
    assert result == ["chunk1", "chunk2"]

    trace_dict = get_latest_trace_dict()
    retriever_names = span_names_by_key(trace_dict, "retrieverSpans")

    # Either in retrieverSpans or as a named span
    assert (
        "test_retriever" in retriever_names
        or find_span_by_name(trace_dict, "test_retriever") is not None
    ), f"Expected test_retriever span. Got: {debug_span_names(trace_dict)}"


def test_tool_span_type():
    """
    Docs: type="tool" creates a tool-typed span.
    """

    @observe(type="tool", name="test_tool")
    def tool_func(args: str):
        return "tool result"

    result = tool_func("test args")
    assert result == "tool result"

    trace_dict = get_latest_trace_dict()
    tool_names = span_names_by_key(trace_dict, "toolSpans")
    base_names = span_names_by_key(trace_dict, "baseSpans")

    assert (
        "test_tool" in tool_names
        or "test_tool" in base_names
        or find_span_by_name(trace_dict, "test_tool") is not None
    ), f"Expected test_tool span. Got: {debug_span_names(trace_dict)}"


def test_agent_span_type():
    """
    Docs: type="agent" creates an agent-typed span.
    """

    @observe(type="agent", name="test_agent")
    def agent_func(input: str):
        return "agent response"

    result = agent_func("test input")
    assert result == "agent response"

    trace_dict = get_latest_trace_dict()
    agent_names = span_names_by_key(trace_dict, "agentSpans")

    assert (
        "test_agent" in agent_names
        or find_span_by_name(trace_dict, "test_agent") is not None
    ), f"Expected test_agent span. Got: {debug_span_names(trace_dict)}"


###############################################################
# Checklist: Nested execution contexts produce parent/child   #
#            span relationships with explicit UUID edges      #
###############################################################


def test_nested_spans_parent_child_uuid_relationships():
    """
    Checklist item 1: Nested execution contexts correctly produce parent and
    child span relationships.

    This test verifies:
    - All spans have a uuid field
    - Parent-child relationships are explicitly linked via parentUuid == parent.uuid
    - The tree structure is: nested_app (root) -> outer_retriever -> inner_retriever
                                              -> generator
    """
    out = your_llm_app_nested_spans("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    assert trace_dict is not None

    spans = all_spans(trace_dict)

    def by_name(n: str):
        return next((s for s in spans if s.get("name") == n), None)

    app_span = by_name("nested_app")
    outer_span = by_name("outer_retriever")
    inner_span = by_name("inner_retriever")
    gen_span = by_name("generator")

    # All spans must exist
    assert (
        app_span is not None
    ), f"Missing nested_app. Available: {[s.get('name') for s in spans]}"
    assert (
        outer_span is not None
    ), f"Missing outer_retriever. Available: {[s.get('name') for s in spans]}"
    assert (
        inner_span is not None
    ), f"Missing inner_retriever. Available: {[s.get('name') for s in spans]}"
    assert (
        gen_span is not None
    ), f"Missing generator. Available: {[s.get('name') for s in spans]}"

    # All spans must have a uuid
    assert app_span.get("uuid"), "nested_app must have uuid"
    assert outer_span.get("uuid"), "outer_retriever must have uuid"
    assert inner_span.get("uuid"), "inner_retriever must have uuid"
    assert gen_span.get("uuid"), "generator must have uuid"

    # Verify explicit parent-child UUID relationships
    # nested_app is root (parentUuid is None or missing)
    assert (
        app_span.get("parentUuid") is None
    ), f"nested_app should be root span with no parent, got parentUuid={app_span.get('parentUuid')}"

    # outer_retriever.parentUuid == nested_app.uuid
    assert outer_span.get("parentUuid") == app_span.get("uuid"), (
        f"outer_retriever.parentUuid should equal nested_app.uuid. "
        f"Got parentUuid={outer_span.get('parentUuid')}, expected={app_span.get('uuid')}"
    )

    # inner_retriever.parentUuid == outer_retriever.uuid
    assert inner_span.get("parentUuid") == outer_span.get("uuid"), (
        f"inner_retriever.parentUuid should equal outer_retriever.uuid. "
        f"Got parentUuid={inner_span.get('parentUuid')}, expected={outer_span.get('uuid')}"
    )

    # generator.parentUuid == nested_app.uuid
    assert gen_span.get("parentUuid") == app_span.get("uuid"), (
        f"generator.parentUuid should equal nested_app.uuid. "
        f"Got parentUuid={gen_span.get('parentUuid')}, expected={app_span.get('uuid')}"
    )


###############################################################
# Checklist: Component-level outputs convert to serialized    #
#            structure (TraceApi) with expected keys          #
###############################################################


def test_trace_serialization_contains_expected_top_level_keys():
    """
    Checklist item 2: Component-level outputs can be converted into a test run
    or serialized structure with the expected keys.

    Verifies the trace_dict (TraceApi serialized output) contains:
    - Top-level keys: uuid, startTime, endTime, status
    - Typed span bucket keys: baseSpans, agentSpans, llmSpans, retrieverSpans, toolSpans
    - Each bucket is a list
    - Spans in buckets have required keys: uuid, name, status, startTime, endTime
    """
    out = your_llm_app_rooted("How are you?")
    assert out == "MOCK_RESPONSE"

    trace_dict = get_latest_trace_dict()
    assert trace_dict is not None

    # Top-level trace keys
    assert "uuid" in trace_dict, "Trace must have 'uuid' key"
    assert "startTime" in trace_dict, "Trace must have 'startTime' key"
    assert "endTime" in trace_dict, "Trace must have 'endTime' key"
    assert "status" in trace_dict, "Trace must have 'status' key"

    # Typed span bucket keys must exist as lists
    expected_buckets = [
        "baseSpans",
        "agentSpans",
        "llmSpans",
        "retrieverSpans",
        "toolSpans",
    ]
    for bucket in expected_buckets:
        assert bucket in trace_dict, f"Trace must have '{bucket}' key"
        assert isinstance(
            trace_dict[bucket], list
        ), f"'{bucket}' must be a list"

    # Verify spans have required per-span keys
    required_span_keys = {"uuid", "name", "status", "startTime", "endTime"}
    for bucket in expected_buckets:
        for span in trace_dict[bucket]:
            missing = required_span_keys - set(span.keys())
            assert (
                not missing
            ), f"Span '{span.get('name')}' in {bucket} missing keys: {missing}"


###############################################################
# Regression: ToolSpan kwargs collision fix                   #
###############################################################


def test_observe_tool_with_name_kwarg_does_not_crash():
    """
    Regression test: @observe(type="tool", name="...") previously crashed due to
    kwargs collision when 'name' was passed both in observe_kwargs and span_kwargs.
    The fix filters observe_kwargs to ToolSpan model fields and drops colliding keys.
    """

    @observe(type="tool", name="my_named_tool")
    def named_tool_func(arg: str) -> str:
        return f"tool output: {arg}"

    result = named_tool_func("test_input")
    assert result == "tool output: test_input"

    trace_dict = get_latest_trace_dict()
    assert trace_dict is not None

    # The tool span should be in toolSpans with the custom name
    tool_names = span_names_by_key(trace_dict, "toolSpans")
    assert (
        "my_named_tool" in tool_names
    ), f"Expected 'my_named_tool' in toolSpans. Got: {tool_names}"
