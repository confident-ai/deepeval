"""ArgumentCorrectness on a traced span with no tool calls of its own judges
the tool calls of the nearest LLM span: the tool's parent when that is an LLM
span, else the latest LLM sibling that started before it."""

import pytest

from deepeval.dataset.golden import Golden
from deepeval.errors import MissingTestCaseParamsError
from deepeval.evaluate.configs import CacheConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.execute import execute_agentic_test_cases_from_loop
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.metrics.argument_correctness.schema import (
    ArgumentCorrectnessScoreReason,
    ArgumentCorrectnessVerdict,
    Verdicts,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import observe
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    EvalSession,
    LlmSpan,
    RetrieverSpan,
    ToolSpan,
    TraceSpanStatus,
)


class RecordingJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.verdict_prompts = []
        super().__init__(model="recording-judge")

    def load_model(self, *args, **kwargs):
        return None

    def generate(self, prompt, schema=None, **kwargs):
        if schema is Verdicts:
            self.verdict_prompts.append(prompt)
            # One verdict per judged tool call, like a real judge; the
            # score denominators count missing verdicts against the metric.
            return Verdicts(
                verdicts=[
                    ArgumentCorrectnessVerdict(verdict="yes")
                    for _ in _judged_tools(prompt)
                ]
            )
        return ArgumentCorrectnessScoreReason(reason="Arguments fit the input.")

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs):
        return "recording-judge"


def _span(cls, name, start, children=(), **fields) -> BaseSpan:
    span = cls(
        uuid=name,
        name=name,
        status=TraceSpanStatus.SUCCESS,
        trace_uuid="trace",
        start_time=start,
        children=list(children),
        **fields,
    )
    for child in span.children:
        child.parent_uuid = span.uuid
    return span


def _traced(span, root_spans, **fields) -> LLMTestCase:
    test_case = LLMTestCase(
        input=fields.pop("input", str(span.input) if span else "trace input"),
        **fields,
    )
    test_case._trace_root_spans = root_spans
    test_case._trace_span = span
    return test_case


def _judged(test_case, async_mode=True) -> str:
    judge = RecordingJudge()
    metric = ArgumentCorrectnessMetric(model=judge, async_mode=async_mode)
    metric.measure(test_case)
    assert len(judge.verdict_prompts) == 1
    return judge.verdict_prompts[0]


def _judged_tools(prompt):
    return [
        name
        for name in (
            "check_application_status",
            "search_agency_kb",
            "lookup_visa_quota",
            "file_report",
            "declared_tool",
        )
        if f'name="{name}"' in prompt
    ]


def _business_sg_trace():
    """An agent whose tools run inside a nested agent's LLM span."""
    llm = _span(
        LlmSpan,
        "generate_response",
        3,
        input="llm prompt",
        children=[
            _span(
                ToolSpan,
                "check_application_status",
                4,
                input={"application_id": "EP-2026-118302"},
            ),
            _span(ToolSpan, "search_agency_kb", 5, input={"query": "EP"}),
        ],
    )
    business_agent = _span(
        AgentSpan,
        "business_sg_agent",
        1,
        input="business agent input",
        children=[
            _span(RetrieverSpan, "retrieve_business_context", 2),
            llm,
        ],
    )
    orchestrator = _span(
        AgentSpan,
        "orchestrate_business_sg_agent",
        0,
        input="What's happening with our EP application?",
        children=[
            business_agent,
            _span(AgentSpan, "business_sg_response_validator", 6),
        ],
    )
    return orchestrator, llm


def _agent_loop_trace():
    """An agent loop: each tool runs beside the LLM span that chose it."""
    plan = _span(LlmSpan, "plan", 1, input="plan input")
    lookup = _span(ToolSpan, "lookup_visa_quota", 2, input={"pass": "EP"})
    answer = _span(LlmSpan, "answer", 3, input="answer input")
    report = _span(ToolSpan, "file_report", 4, input={"case": "EP-1"})
    agent = _span(
        AgentSpan,
        "agent",
        0,
        input="agent input",
        children=[report, answer, lookup, plan],  # end order, not start order
    )
    return agent, plan, lookup, answer, report


@pytest.mark.parametrize("async_mode", [True, False])
def test_agent_span_judges_tool_calls_of_nested_llm_spans(async_mode):
    orchestrator, _ = _business_sg_trace()

    prompt = _judged(_traced(orchestrator, [orchestrator]), async_mode)

    assert _judged_tools(prompt) == [
        "check_application_status",
        "search_agency_kb",
    ]
    assert "What's happening with our EP application?" in prompt
    assert '"application_id": "EP-2026-118302"' in prompt


def test_llm_span_judges_its_sibling_tool_calls():
    agent, plan, _, answer, _ = _agent_loop_trace()

    assert _judged_tools(_judged(_traced(plan, [agent]))) == [
        "lookup_visa_quota"
    ]
    assert _judged_tools(_judged(_traced(answer, [agent]))) == ["file_report"]


def test_llm_span_judges_its_child_tool_calls():
    orchestrator, llm = _business_sg_trace()

    assert _judged_tools(_judged(_traced(llm, [orchestrator]))) == [
        "check_application_status",
        "search_agency_kb",
    ]


def test_tool_span_is_judged_against_the_llm_span_before_it():
    agent, _, _, _, report = _agent_loop_trace()

    prompt = _judged(_traced(report, [agent]))

    assert _judged_tools(prompt) == ["file_report"]
    assert "answer input" in prompt
    assert "plan input" not in prompt


def test_tool_span_is_judged_against_its_llm_parent():
    orchestrator, llm = _business_sg_trace()
    tool = llm.children[1]

    prompt = _judged(_traced(tool, [orchestrator]))

    assert _judged_tools(prompt) == ["search_agency_kb"]
    assert "llm prompt" in prompt


def test_trace_level_judges_every_llm_owned_tool_call():
    agent, *_ = _agent_loop_trace()

    prompt = _judged(_traced(None, [agent]))

    assert _judged_tools(prompt) == ["lookup_visa_quota", "file_report"]
    assert "trace input" in prompt


def test_tool_calls_the_llm_span_lists_itself_are_not_duplicated():
    orchestrator, llm = _business_sg_trace()
    llm.tools_called = [ToolCall(name="check_application_status")]

    prompt = _judged(_traced(orchestrator, [orchestrator]))

    assert prompt.count('name="check_application_status"') == 1
    assert _judged_tools(prompt) == ["check_application_status"]


def test_the_test_cases_own_tool_calls_win():
    agent, plan, *_ = _agent_loop_trace()
    test_case = _traced(
        plan, [agent], tools_called=[ToolCall(name="declared_tool")]
    )

    assert _judged_tools(_judged(test_case)) == ["declared_tool"]


def test_resolution_does_not_change_the_shared_test_case():
    agent, _, _, _, report = _agent_loop_trace()
    test_case = _traced(report, [agent])

    _judged(test_case)

    assert test_case.tools_called is None
    assert test_case.input == str(report.input)


@pytest.mark.parametrize("async_mode", [True, False])
def test_tool_call_without_an_llm_span_is_still_missing(async_mode):
    tool = _span(ToolSpan, "lookup_visa_quota", 1, input={"pass": "EP"})
    agent = _span(
        AgentSpan,
        "agent",
        0,
        input="agent input",
        children=[tool, _span(LlmSpan, "summarize", 2, input="x")],
    )
    metric = ArgumentCorrectnessMetric(
        model=RecordingJudge(), async_mode=async_mode
    )

    with pytest.raises(MissingTestCaseParamsError):
        metric.measure(_traced(agent, [agent]))


def test_untraced_test_case_without_tool_calls_is_still_missing():
    metric = ArgumentCorrectnessMetric(model=RecordingJudge())

    with pytest.raises(MissingTestCaseParamsError):
        metric.measure(LLMTestCase(input="What's the minimum salary?"))


@pytest.fixture
def _offline_tracing(monkeypatch):
    monkeypatch.setenv("CONFIDENT_TRACE_FLUSH", "0")
    monkeypatch.setattr(trace_manager, "tracing_enabled", False)
    yield
    trace_manager.eval_session = EvalSession()


def test_executor_gives_an_agent_span_metric_the_span_tree(_offline_tracing):
    judge = RecordingJudge()
    metric = ArgumentCorrectnessMetric(model=judge, async_mode=False)

    @observe(type="tool")
    def check_application_status(application_id: str):
        return "pending"

    @observe(type="tool")
    def search_agency_kb(query: str):
        return "EP guidance"

    @observe(type="llm", model="gpt-4o-mini")
    def generate_response(prompt: str):
        check_application_status(application_id="EP-2026-118302")
        search_agency_kb(query="EP processing time")
        return "Your EP application is pending."

    @observe(type="agent")
    def business_sg_agent(message: str):
        return generate_response(prompt=message)

    @observe(type="agent", metrics=[metric])
    def orchestrate_business_sg_agent(message: str):
        return business_sg_agent(message=message)

    evaluation = execute_agentic_test_cases_from_loop(
        goldens=[Golden(input="What's happening with our EP application?")],
        trace_metrics=None,
        test_results=[],
        display_config=DisplayConfig(show_indicator=False, verbose_mode=False),
        cache_config=CacheConfig(write_cache=False),
        error_config=ErrorConfig(
            ignore_errors=False, skip_on_missing_params=False
        ),
        _use_bar_indicator=False,
    )
    for golden in evaluation:
        orchestrate_business_sg_agent(message=golden.input)

    assert metric.error is None
    assert metric.score == 1.0
    assert len(judge.verdict_prompts) == 1
    prompt = judge.verdict_prompts[0]
    assert _judged_tools(prompt) == [
        "check_application_status",
        "search_agency_kb",
    ]
    assert "What's happening with our EP application?" in prompt
