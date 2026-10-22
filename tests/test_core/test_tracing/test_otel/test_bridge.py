"""Which confident-trace spans ``DeepEvalSpanBridge`` keeps, and how."""

import asyncio
import json

import pytest
from opentelemetry.trace import set_span_in_context

from deepeval.config.settings import reset_settings
from deepeval.contextvars import set_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ToolCall
from deepeval.tracing import (
    next_llm_span,
    observe,
    trace,
    update_current_span,
    update_current_trace,
    update_llm_span,
)
from deepeval.tracing.otel.setup import (
    ensure_confident_trace,
)
from deepeval.tracing.trace_context import LlmSpanContext
from deepeval.tracing.trace_test_manager import trace_testing_manager
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import (
    LlmSpan,
    TraceSpanStatus,
)

ct = pytest.importorskip("confident_trace")

from .helpers import (  # noqa: E402
    MEASURED,
    RecordingMetric,
    exported,
    names,
    names_of,
)


class _JudgeModel(DeepEvalBaseLLM):
    """Stands in for a judge model whose provider confident-trace traces."""

    def load_model(self):
        return None

    def generate(self, prompt):
        with ct.span("judge call", type="llm"):
            return "ok"

    async def a_generate(self, prompt):
        return self.generate(prompt)

    def get_model_name(self):
        return "judge"


################ Ownership ################


def test_spans_outside_deepeval_are_left_to_confident_trace(exporter, posted):
    with ct.span("bare"):
        pass

    assert exported(exporter) == ["bare"]
    assert posted == []
    assert trace_manager.active_traces == {}


def test_observe_owns_the_spans_inside_it(exporter, posted):
    @observe(type="agent", name="agent")
    def agent():
        with ct.span("plan"):
            with ct.span("generate", type="llm", model="gpt-4o"):
                pass
        return "done"

    agent()

    assert exported(exporter) == []
    (collected,) = posted
    (root,) = collected.root_spans
    assert root.name == "agent"
    (plan,) = root.children
    assert (plan.trace_uuid, plan.parent_uuid) == (collected.uuid, root.uuid)
    (generate,) = plan.children
    assert isinstance(generate, LlmSpan)
    assert generate.model == "gpt-4o"
    # A nested span's entry fields describe it, not the deepeval trace.
    assert (collected.name, collected.output) == (None, "done")


def test_a_root_span_fills_only_unset_trace_fields(exporter, posted):
    @ct.span(name="answer")
    def answer(question):
        return question.upper()

    with trace():
        answer("q")
    with trace(name="mine"):
        answer("q")

    first, second = posted
    assert (first.name, first.input, first.output) == (
        "answer",
        {"args": ["q"], "kwargs": {}},
        "Q",
    )
    assert second.name == "mine"


def test_a_later_model_call_in_a_deepeval_trace_sets_its_output(tracer, posted):
    def generate(answer):
        reply = [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": answer}],
            }
        ]
        with tracer.start_as_current_span(
            "chat",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.output.messages": json.dumps(reply),
            },
        ):
            pass

    # Each call is a trace of its own to OTel, but one deepeval trace.
    @observe(name="endpoint")
    def endpoint():
        generate("thinking")
        generate("sunny")

    endpoint()

    assert posted[0].output == "sunny"


def test_span_defaults_staged_in_deepeval(exporter, posted):
    from_context, staged = RecordingMetric(), RecordingMetric()
    context = LlmSpanContext(metrics=[from_context], expected_output="x")

    with trace(llm_span_context=context):
        with next_llm_span(metrics=[staged], metric_collection="collection"):
            with ct.span("generate", type="llm"):
                pass

    (generate,) = posted[0].root_spans
    # ``next_*_span`` is more specific than the trace's span context.
    assert generate.metrics == [staged]
    assert generate.expected_output == "x"
    assert generate.metric_collection == "collection"


################ Tree building ################


def test_children_attach_to_a_parent_that_ended_first(tracer, posted):
    with trace():
        parent = tracer.start_span(
            "parent", attributes={"gen_ai.operation.name": "invoke_agent"}
        )
        child = tracer.start_span(
            "child",
            context=set_span_in_context(parent),
            attributes={"gen_ai.operation.name": "chat"},
        )
        parent.end()
        child.end()

    (parent,) = posted[0].root_spans
    assert names(parent.children) == ["child"]


def test_non_ai_spans_stay_only_as_ancestors(tracer, posted):
    with trace():
        with tracer.start_as_current_span("request"):
            with tracer.start_as_current_span("http"):
                pass
            with ct.span("generate", type="llm"):
                pass
        with tracer.start_as_current_span("cleanup"):
            pass

    (request,) = posted[0].root_spans
    assert names(request.children) == ["generate"]


def test_children_of_a_dropped_span_move_to_its_parent(tracer, posted):
    with trace():
        with ct.span("agent", type="agent"):
            wrapper = tracer.start_span("wrapper")
            child = tracer.start_span(
                "generate",
                context=set_span_in_context(wrapper),
                attributes={"gen_ai.operation.name": "chat"},
            )
            wrapper.end()
            child.end()

    (agent,) = posted[0].root_spans
    assert names(agent.children) == ["generate"]


################ Traces the bridge owns ################


def test_async_evaluation_collects_one_trace_per_golden(
    exporter, async_evaluation
):
    goldens = [Golden(input="first"), Golden(input="second")]

    @ct.span(name="app")
    async def app(question):
        await asyncio.sleep(0)
        with ct.span("generate", type="llm"):
            await asyncio.sleep(0)
        return question.upper()

    async def run(golden):
        set_current_golden(golden)
        return await app(golden.input)

    async def interleaved():
        await asyncio.gather(*(run(golden) for golden in goldens))

    asyncio.run(interleaved())

    assert exported(exporter) == []
    traces = async_evaluation.traces_to_evaluate
    assert len(traces) == 2
    for collected in traces:
        golden = async_evaluation.trace_uuid_to_golden[collected.uuid]
        assert collected.output == golden.input.upper()
        (root,) = collected.root_spans
        assert names(collected.root_spans) == ["app"]
        assert names(root.children) == ["generate"]


def test_trace_fields_merge_depth_first_with_later_spans_winning(
    tracer, async_evaluation
):
    with tracer.start_as_current_span(
        "root",
        attributes={
            "confident.trace.name": "root",
            "confident.trace.user_id": "user",
        },
    ):
        with tracer.start_as_current_span(
            "child", attributes={"confident.trace.name": "child"}
        ):
            pass

    (collected,) = async_evaluation.traces_to_evaluate
    assert (collected.name, collected.user_id) == ("child", "user")


def test_an_errored_root_errors_the_trace(exporter, async_evaluation):
    with pytest.raises(ValueError):
        with ct.span("boom"):
            raise ValueError("secret")

    (collected,) = async_evaluation.traces_to_evaluate
    assert collected.status == TraceSpanStatus.ERRORED


def test_a_trace_of_only_non_ai_spans_is_discarded(tracer, async_evaluation):
    with tracer.start_as_current_span("http"):
        pass

    assert async_evaluation.traces_to_evaluate == []
    assert async_evaluation.pending_traces == {}
    assert trace_manager.active_traces == {}


def test_trace_shape_tests_capture_bare_spans(exporter, monkeypatch):
    monkeypatch.setattr(trace_testing_manager, "test_name", "bare")
    monkeypatch.setattr(trace_testing_manager, "test_dict", None)

    with ct.span("bare", type="agent"):
        pass

    assert exported(exporter) == []
    assert names_of(trace_testing_manager.test_dict["agentSpans"]) == ["bare"]


################ Traces from framework-native spans ################


def native_agent_run(tracer, conversation="thread"):
    """An agent that calls a model and a tool, with no ``confident.trace.*``."""
    messages = [{"role": "user", "parts": [{"type": "text", "content": "q"}]}]
    answer = [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "a"}],
            "finish_reason": "stop",
        }
    ]
    with tracer.start_as_current_span(
        "invoke_agent planner",
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "planner",
            "gen_ai.conversation.id": conversation,
        },
    ):
        with tracer.start_as_current_span("step"):
            with tracer.start_as_current_span(
                "execute_tool lookup",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "lookup",
                    "gen_ai.tool.call.arguments": '{"city": "Paris"}',
                    "gen_ai.tool.call.result": "sunny",
                },
            ):
                pass
        with tracer.start_as_current_span(
            "chat",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.input.messages": json.dumps(messages),
                "gen_ai.output.messages": json.dumps(answer),
            },
        ):
            pass


def test_an_owned_native_trace_gets_its_content_from_its_spans(
    tracer, async_evaluation
):
    native_agent_run(tracer)

    (collected,) = async_evaluation.traces_to_evaluate
    assert (collected.name, collected.input, collected.output) == (
        "planner",
        "q",
        "a",
    )
    assert collected.thread_id == "thread"
    assert collected.tools_called == [
        ToolCall(
            name="lookup", input_parameters={"city": "Paris"}, output="sunny"
        )
    ]
    (agent,) = collected.root_spans
    # The agent lists the tool its intermediate step ran.
    assert [call.name for call in agent.tools_called] == ["lookup"]


def test_a_deepeval_trace_fills_only_what_its_code_left_unset(tracer, posted):
    with trace(output="mine"):
        native_agent_run(tracer)

    (collected,) = posted
    assert (collected.input, collected.output) == ("q", "mine")
    assert collected.thread_id == "thread"
    assert [call.name for call in collected.tools_called] == ["lookup"]


################ deepeval code inside collected spans ################


def test_span_updates_override_what_a_collected_span_recorded(exporter, posted):
    staged = RecordingMetric()
    with trace():
        with ct.span("generate", type="llm", model="recorded"):
            update_current_span(
                name="renamed", metadata={"from": "deepeval"}, metrics=[staged]
            )
            update_llm_span(model="gpt-4o")

    (generate,) = posted[0].root_spans
    assert (generate.name, generate.model) == ("renamed", "gpt-4o")
    assert generate.metadata == {"from": "deepeval"}
    assert generate.metrics == [staged]


def test_observe_nests_inside_collected_spans_and_back(exporter, posted):
    @observe(name="helper")
    def helper():
        update_current_span(metadata={"from": "helper"})
        with ct.span("inner", type="tool"):
            pass
        return "done"

    with trace():
        with ct.span("outer", type="agent"):
            helper()

    (outer,) = posted[0].root_spans
    (helper_span,) = outer.children
    assert (helper_span.name, helper_span.metadata) == (
        "helper",
        {"from": "helper"},
    )
    assert names(helper_span.children) == ["inner"]
    assert outer.metadata is None


def test_trace_updates_win_over_what_spans_recorded(exporter, async_evaluation):
    @ct.span(name="app")
    def app():
        update_current_trace(output="mine", tags=["deepeval"])
        return "recorded"

    app()

    (collected,) = async_evaluation.traces_to_evaluate
    assert (collected.output, collected.tags) == ("mine", ["deepeval"])


def test_updates_inside_exported_spans_reach_confident_trace(exporter, posted):
    with ct.span("bare"):
        update_current_span(
            metadata={"from": "deepeval"},
            tools_called=[ToolCall(name="lookup")],
            metrics=[RecordingMetric()],
        )
        update_current_trace(name="renamed", thread_id="thread")

    ct.flush()
    (bare,) = exporter.get_finished_spans()
    assert json.loads(bare.attributes["confident.span.metadata"]) == {
        "from": "deepeval"
    }
    assert json.loads(bare.attributes["confident.span.tools_called"]) == [
        {"name": "lookup", "type": "FUNCTION"}
    ]
    assert bare.attributes["confident.trace.name"] == "renamed"
    assert bare.attributes["confident.trace.thread_id"] == "thread"
    assert posted == []


################ deepeval's own model calls ################


def test_judge_model_calls_are_not_traced(exporter, async_evaluation):
    judge = _JudgeModel(model="judge")
    judge.generate("score this")
    asyncio.run(judge.a_generate("score this"))

    assert exported(exporter) == []
    assert async_evaluation.traces_to_evaluate == []
    assert trace_manager.active_traces == {}


def test_internal_tracing_opts_judge_calls_back_in(exporter, monkeypatch):
    monkeypatch.setenv("CONFIDENT_TRACE_INTERNAL", "1")
    reset_settings(reload_dotenv=False)

    _JudgeModel(model="judge").generate("score this")

    assert exported(exporter) == ["judge call"]


################ End to end ################


def generate(question):
    with next_llm_span(metrics=[RecordingMetric()]):
        with ct.span("generate", type="llm"):
            ct.update_span(input=question, output=question.upper())


async def a_generate(question):
    with next_llm_span(metrics=[RecordingMetric()]):
        with ct.span("generate", type="llm"):
            await asyncio.sleep(0)
            ct.update_span(input=question, output=question.upper())


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
def test_evals_iterator_scores_confident_trace_spans(exporter, run_async):
    dataset = EvaluationDataset(goldens=[Golden(input="hi")])
    for golden in dataset.evals_iterator(
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        cache_config=CacheConfig(write_cache=False),
    ):
        if run_async:
            dataset.evaluate(a_generate(golden.input))
        else:
            generate(golden.input)

    assert exported(exporter) == []
    assert [(case.input, case.actual_output) for case in MEASURED] == [
        ("hi", "HI")
    ]


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
def test_trace_metrics_score_a_bare_openai_call(
    exporter, openai_client, run_async
):
    ensure_confident_trace(("openai",))
    client = openai_client("text", asynchronous=run_async)

    dataset = EvaluationDataset(goldens=[Golden(input="weather?")])
    for golden in dataset.evals_iterator(
        metrics=[RecordingMetric()],
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        cache_config=CacheConfig(write_cache=False),
    ):
        call = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": golden.input}],
        )
        if run_async:
            dataset.evaluate(call)

    assert exported(exporter) == []
    # The call gives its trace the output that trace-level metrics score.
    assert [(case.input, case.actual_output) for case in MEASURED] == [
        ("weather?", "sunny")
    ]


def test_flushing_traces_flushes_exported_spans(tracer, exporter):
    from deepeval.tracing import flush_traces

    with tracer.start_as_current_span(
        "chat", attributes={"gen_ai.operation.name": "chat"}
    ):
        pass

    assert flush_traces(timeout=5)
    assert [span.name for span in exporter.get_finished_spans()] == ["chat"]
