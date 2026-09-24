"""Every metric wired for eval modes, checked under each mode:

(a) ``system_one``: Jev runs the whole metric in one request, no LLM call.
(b) ``hybrid``: the LLM extracts, Jev takes the decisions, and the LLM
    decision prompt is never sent.
(c) ``hybrid`` with Jev down: falls back to the LLM and records why.
(d) ``llm``: Jev is never called.

Each runs through both ``measure`` (``async_mode=False``) and ``a_measure``."""

import importlib.util
import json
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from pydantic import BaseModel

from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ConversationCompletenessMetric,
    GoalAccuracyMetric,
    JsonCorrectnessMetric,
    KnowledgeRetentionMetric,
    MCPTaskCompletionMetric,
    MCPUseMetric,
    MultiTurnMCPUseMetric,
    PlanAdherenceMetric,
    PlanQualityMetric,
    RoleAdherenceMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    ToolUseMetric,
    TopicAdherenceMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,
    TurnRelevancyMetric,
)
from deepeval.metrics.community.citation_faithfulness.citation_faithfulness import (
    CitationFaithfulnessMetric,
)
from deepeval.models.system_one.schema import ChoiceAnswer
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MCPServer,
    MCPToolCall,
    ToolCall,
    Turn,
)
from tests.test_metrics.system_one_fakes import (
    CannedLLM,
    ExplodingLLM,
    ExplodingSystemOneModel,
    FakeSystemOneModel,
    answer_everything,
)

###############################################
# Fakes and fixtures
###############################################


class RoutingLLM(CannedLLM):
    """Replies to a prompt with the reply of the first route whose key is in
    it, else with ``default``."""

    def __init__(self, routes: List[Tuple[str, str]], default: str):
        super().__init__(default, name="routing-llm")
        self.routes = routes

    def generate(self, prompt, *args, **kwargs):
        text = str(prompt)
        self.prompts.append(text)
        return next((r for key, r in self.routes if key in text), self.reply)

    def was_sent(self, key: str) -> bool:
        return any(key in p for p in self.prompts)


def jev() -> FakeSystemOneModel:
    return FakeSystemOneModel(answer_fn=answer_everything())


def jev_down() -> ExplodingSystemOneModel:
    return ExplodingSystemOneModel(ConnectionError("down"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("DEEPEVAL_MODE", "DEEPEVAL_EVAL_MODE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPEVAL_DISABLE_LEGACY_KEYFILE", "1")
    yield


@pytest.fixture(autouse=True)
def _mcp_types(monkeypatch):
    """Minimal ``mcp.types`` when the ``mcp`` package isn't installed: the MCP
    test cases only type-check against these classes."""
    if importlib.util.find_spec("mcp") is not None:
        return

    class Tool(BaseModel):
        name: str
        description: str
        inputSchema: dict

    class Resource(BaseModel):
        uri: str

    class Prompt(BaseModel):
        name: str

    class CallToolResult(BaseModel):
        content: list
        structuredContent: dict

    class ReadResourceResult(BaseModel):
        contents: list = []

    class GetPromptResult(BaseModel):
        messages: list = []

    mcp, mcp_types = types.ModuleType("mcp"), types.ModuleType("mcp.types")
    for cls in (
        Tool,
        Resource,
        Prompt,
        CallToolResult,
        ReadResourceResult,
        GetPromptResult,
    ):
        setattr(mcp_types, cls.__name__, cls)
    mcp.types = mcp_types
    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.types", mcp_types)


async def measure(metric, test_case, is_async: bool) -> float:
    if is_async:
        return await metric.a_measure(test_case, _show_indicator=False)
    return metric.measure(test_case, _show_indicator=False)


###############################################
# Test cases
###############################################

TRACE = {
    "name": "agent",
    "type": "agent",
    "input": {"input": "Book a flight to Paris"},
    "output": "Booked",
    "tokens": 123,
    "children": [
        {
            "name": "search_flights",
            "type": "tool",
            "input": {"dest": "Paris"},
            "output": ["AF1"],
        }
    ],
}
TOOLS = [ToolCall(name="get_weather", description="Weather for a city")]


def traced_tc(trace: bool = True) -> LLMTestCase:
    tc = LLMTestCase(
        input="Book a flight to Paris",
        actual_output="Booked AF1",
        tools_called=[ToolCall(name="get_weather")],
        expected_tools=[ToolCall(name="get_weather")],
    )
    if trace:
        tc._trace_dict = TRACE
    return tc


def rag_tc() -> LLMTestCase:
    return LLMTestCase(
        input="What did Einstein win?",
        actual_output="Einstein won the Nobel Prize [1]. He was born in Ulm [2].",
        expected_output="Einstein won the Nobel Prize. He was born in Ulm.",
        retrieval_context=[
            "Einstein won the Nobel Prize. There was a cat.",
            "Einstein was born in Ulm.",
        ],
    )


def rag_conv_tc() -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="What did Einstein win?"),
            Turn(
                role="assistant",
                content="The Nobel Prize.",
                retrieval_context=[
                    "Einstein won the Nobel Prize. There was a cat."
                ],
            ),
            Turn(role="user", content="Where was he born?"),
            Turn(
                role="assistant",
                content="Ulm.",
                retrieval_context=["Einstein was born in Ulm."],
            ),
        ],
        expected_outcome="Einstein won the Nobel Prize. He was born in Ulm.",
    )


def support_conv_tc() -> ConversationalTestCase:
    return ConversationalTestCase(
        chatbot_role="A polite airline support agent",
        turns=[
            Turn(
                role="user",
                content="Hi, I'm Ann. I want to book a flight to Paris.",
            ),
            Turn(
                role="assistant",
                content="Sure Ann, when would you like to fly?",
            ),
            Turn(
                role="user",
                content="Next Monday. Also can I get a refund for my last trip?",
            ),
            Turn(
                role="assistant",
                content="Booked for Monday. Your refund is on its way.",
            ),
        ],
    )


def tool_conv_tc() -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Weather in Paris?"),
            Turn(
                role="assistant",
                content="It's sunny.",
                tools_called=[
                    ToolCall(
                        name="get_weather", input_parameters={"city": "Paris"}
                    )
                ],
            ),
        ]
    )


def _mcp_server_and_call() -> Tuple[MCPServer, MCPToolCall]:
    from mcp.types import CallToolResult, Tool

    tool = Tool(
        name="get_weather",
        description="Get weather",
        inputSchema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    )
    call = MCPToolCall(
        name="get_weather",
        args={"city": "Paris"},
        result=CallToolResult(
            content=[{"type": "text", "text": "sunny"}],
            structuredContent={"result": "sunny"},
        ),
    )
    return MCPServer(server_name="weather", available_tools=[tool]), call


def mcp_tc() -> LLMTestCase:
    server, call = _mcp_server_and_call()
    return LLMTestCase(
        input="Weather in Paris?",
        actual_output="It's sunny.",
        mcp_servers=[server],
        mcp_tools_called=[call],
    )


def mcp_conv_tc() -> ConversationalTestCase:
    server, call = _mcp_server_and_call()
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Weather in Paris?"),
            Turn(role="assistant", content="Checking", mcp_tools_called=[call]),
            Turn(role="assistant", content="It's sunny."),
        ],
        mcp_servers=[server],
    )


###############################################
# Metric cases
###############################################

REASON = json.dumps({"reason": "LLM reason."})
SCORE = json.dumps({"score": 0.4, "reason": "LLM reason."})
PLAN = json.dumps(
    {
        "task": "Book a flight",
        "plan": ["search", "book"],
        "score": 0.4,
        "reason": "r",
    }
)


def no_trace_state(state: Dict[str, Any]) -> None:
    assert "tokens" not in state["trace"]


def tool_use_state(state: Dict[str, Any]) -> None:
    assert state["available_tools"][0]["name"] == "get_weather"
    assert state["turns"][1]["tools_called"]


@dataclass
class Case:
    id: str
    metric: Callable[..., Any]
    test_case: Callable[[], Any]
    # substring of the LLM decision prompt Jev replaces under hybrid
    decision_key: str
    # score from the LLM chain (the routes answer every decision negatively)
    llm_score: float
    routes: List[Tuple[str, str]] = field(default_factory=list)
    default_reply: str = REASON
    # one Jev request per decision unit (node, turn, intention, ...)
    hybrid_jev_calls: int = 1
    # score when Jev answers every question at its most positive
    hybrid_score: float = 1.0
    check_state: Optional[Callable[[Dict[str, Any]], None]] = None
    # False when the Jev reason is one part of a composite reason
    reason_is_jev: bool = True

    def llm(self) -> RoutingLLM:
        return RoutingLLM(self.routes, self.default_reply)


CASES = [
    Case(
        "StepEfficiency",
        StepEfficiencyMetric,
        traced_tc,
        decision_key="**efficiency auditor**",
        default_reply=PLAN,
        llm_score=0.4,
        check_state=no_trace_state,
    ),
    Case(
        "PlanAdherence",
        PlanAdherenceMetric,
        traced_tc,
        decision_key="**adversarial plan adherence evaluator**",
        default_reply=PLAN,
        llm_score=0.4,
        check_state=no_trace_state,
    ),
    Case(
        "PlanQuality",
        PlanQualityMetric,
        traced_tc,
        decision_key="**plan quality evaluator**",
        default_reply=PLAN,
        llm_score=0.4,
        check_state=no_trace_state,
    ),
    Case(
        "ToolUse",
        lambda **kw: ToolUseMetric(available_tools=TOOLS, **kw),
        tool_conv_tc,
        decision_key="**Tool Selection Quality**",
        default_reply=PLAN,
        llm_score=0.4,
        hybrid_jev_calls=2,
        check_state=tool_use_state,
    ),
    Case(
        "ContextualRelevancy",
        ContextualRelevancyMetric,
        rag_tc,
        decision_key="Based on the input and context, please generate",
        routes=[
            (
                "Based on the input and context, please generate",
                json.dumps({"verdicts": [{"statement": "s", "verdict": "no"}]}),
            )
        ],
        llm_score=0.0,
        hybrid_jev_calls=2,
    ),
    Case(
        "TurnRelevancy",
        TurnRelevancyMetric,
        rag_conv_tc,
        decision_key="Based on the given list of message exchanges",
        routes=[
            (
                "Based on the given list of message exchanges",
                json.dumps({"verdict": "no", "reason": "r"}),
            )
        ],
        llm_score=0.0,
        hybrid_jev_calls=2,
    ),
    Case(
        "TurnContextualRecall",
        TurnContextualRecallMetric,
        rag_conv_tc,
        decision_key="For EACH sentence in the given assistant output",
        routes=[
            (
                "For EACH sentence in the given assistant output",
                json.dumps({"verdicts": [{"verdict": "no", "reason": "r"}]}),
            )
        ],
        llm_score=0.0,
        hybrid_jev_calls=2,
    ),
    Case(
        "TurnContextualRelevancy",
        TurnContextualRelevancyMetric,
        rag_conv_tc,
        decision_key="Based on the user message and context, please generate",
        routes=[
            (
                "Based on the user message and context, please generate",
                json.dumps({"verdicts": [{"statement": "s", "verdict": "no"}]}),
            )
        ],
        llm_score=0.0,
        hybrid_jev_calls=3,
    ),
    Case(
        "CitationFaithfulness",
        CitationFaithfulnessMetric,
        rag_tc,
        decision_key="You are a faithfulness judge",
        routes=[
            (
                "You are a faithfulness judge",
                json.dumps({"verdict": "unfaithful", "reasoning": "r"}),
            )
        ],
        llm_score=0.0,
        hybrid_jev_calls=1,
    ),
    Case(
        "ConversationCompleteness",
        ConversationCompletenessMetric,
        support_conv_tc,
        decision_key="whether given user intention was satisfied",
        routes=[
            (
                "extract all user intentions",
                json.dumps({"intentions": ["book a flight", "get a refund"]}),
            ),
            (
                "whether given user intention was satisfied",
                json.dumps({"verdict": "no", "reason": "x"}),
            ),
        ],
        llm_score=0.0,
        hybrid_jev_calls=2,
    ),
    Case(
        "KnowledgeRetention",
        KnowledgeRetentionMetric,
        support_conv_tc,
        decision_key="**contradicts** or **forgets**",
        routes=[
            (
                "extract **only the factual information",
                json.dumps({"data": {"Name": "Ann"}}),
            ),
            ("**contradicts** or **forgets**", json.dumps({"verdict": "no"})),
        ],
        llm_score=1.0,
        hybrid_jev_calls=2,
        # a Noul "yes" here means the assistant forgot, so a confident Jev
        # scores retention 0
        hybrid_score=0.0,
    ),
    Case(
        "RoleAdherence",
        RoleAdherenceMetric,
        support_conv_tc,
        decision_key="did not adhere to the specified chatbot role",
        routes=[
            (
                "did not adhere to the specified chatbot role",
                json.dumps({"verdicts": [{"index": 1, "reason": "x"}]}),
            )
        ],
        llm_score=0.5,
        hybrid_jev_calls=2,
    ),
    Case(
        "GoalAccuracy",
        GoalAccuracyMetric,
        support_conv_tc,
        decision_key="**goal accuracy**",
        routes=[
            ("**goal accuracy**", SCORE),
            ("**planning quality**", SCORE),
        ],
        default_reply="final llm reason",
        llm_score=0.4,
        hybrid_jev_calls=4,
    ),
    Case(
        "TopicAdherence",
        lambda **kw: TopicAdherenceMetric(
            relevant_topics=["travel booking", "refunds"], **kw
        ),
        support_conv_tc,
        decision_key="four possible verdicts",
        routes=[
            (
                "extract question-answer (QA) pairs",
                json.dumps(
                    {
                        "qa_pairs": [
                            {"question": "q1", "response": "r1"},
                            {"question": "q2", "response": "r2"},
                        ]
                    }
                ),
            ),
            (
                "four possible verdicts",
                json.dumps({"verdict": "FN", "reason": "x"}),
            ),
        ],
        llm_score=0.0,
        hybrid_jev_calls=8,
    ),
    Case(
        "MCPUse",
        MCPUseMetric,
        mcp_tc,
        decision_key="Evaluate whether the tools (primitives) selected",
        default_reply=SCORE,
        llm_score=0.4,
        hybrid_jev_calls=2,
    ),
    Case(
        "MultiTurnMCPUse",
        MultiTurnMCPUseMetric,
        mcp_conv_tc,
        decision_key="Evaluate whether the tools, resources, and prompts",
        default_reply=SCORE,
        llm_score=0.4,
        hybrid_jev_calls=2,
    ),
    Case(
        "MCPTaskCompletion",
        MCPTaskCompletionMetric,
        mcp_conv_tc,
        decision_key="Evaluate whether the user's task has been successfully",
        default_reply=SCORE,
        llm_score=0.4,
    ),
    Case(
        # Jev only takes the available_tools selection score
        "ToolCorrectness",
        lambda **kw: ToolCorrectnessMetric(available_tools=TOOLS, **kw),
        traced_tc,
        decision_key="assessing the **Tool Selection** quality",
        default_reply=SCORE,
        llm_score=0.4,
        reason_is_jev=False,
    ),
    Case(
        "ContextualRecall",
        ContextualRecallMetric,
        rag_tc,
        decision_key="For EACH sentence in the given expected output",
        routes=[
            (
                "For EACH sentence in the given expected output",
                json.dumps({"verdicts": [{"verdict": "no", "reason": "r"}]}),
            )
        ],
        llm_score=0.0,
    ),
    Case(
        "TaskCompletion",
        TaskCompletionMetric,
        traced_tc,
        decision_key="Given the task (desired outcome) and the actual achieved",
        routes=[
            (
                "Given a nested workflow trace",
                json.dumps({"task": "Book a flight", "outcome": "Booked AF1"}),
            ),
            (
                "Given the task (desired outcome) and the actual achieved",
                json.dumps({"verdict": 0.4, "reason": "r"}),
            ),
        ],
        llm_score=0.4,
        check_state=no_trace_state,
    ),
]

cases = pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
sync_and_async = pytest.mark.parametrize(
    "is_async", [False, True], ids=["sync", "async"]
)


###############################################
# (a) - (d) for every metric
###############################################


@cases
@sync_and_async
@pytest.mark.asyncio
async def test_system_one_decides_in_one_request(case: Case, is_async):
    reasons = []
    for _ in range(2):
        model = jev()
        metric = case.metric(
            model=ExplodingLLM(),
            system_one_model=model,
            eval_mode="system_one",
            async_mode=is_async,
        )
        score = await measure(metric, case.test_case(), is_async)
        assert 0 <= score <= 1
        assert len(model.calls) == 1
        if case.reason_is_jev:
            assert metric.reason.startswith("Decided by fake-jev")
        else:
            assert "Decided by fake-jev" in metric.reason
        reasons.append(metric.reason)
    assert reasons[0] == reasons[1]
    if case.check_state:
        case.check_state(model.calls[0][0])


@cases
@sync_and_async
@pytest.mark.asyncio
async def test_hybrid_jev_takes_the_decisions(case: Case, is_async):
    model, llm = jev(), case.llm()
    metric = case.metric(
        model=llm,
        system_one_model=model,
        eval_mode="hybrid",
        async_mode=is_async,
    )
    score = await measure(metric, case.test_case(), is_async)
    assert len(model.calls) == case.hybrid_jev_calls
    assert not llm.was_sent(case.decision_key)
    assert score == pytest.approx(case.hybrid_score)
    assert metric.system_one_fallback_reason is None
    assert metric.reason


@cases
@sync_and_async
@pytest.mark.asyncio
async def test_hybrid_falls_back_to_llm_when_jev_fails(case: Case, is_async):
    model, llm = jev_down(), case.llm()
    metric = case.metric(
        model=llm,
        system_one_model=model,
        eval_mode="hybrid",
        async_mode=is_async,
    )
    score = await measure(metric, case.test_case(), is_async)
    assert model.calls
    assert llm.was_sent(case.decision_key)
    assert score == pytest.approx(case.llm_score)
    assert "down" in metric.system_one_fallback_reason


@cases
@sync_and_async
@pytest.mark.asyncio
async def test_llm_mode_never_calls_jev(case: Case, is_async):
    model, llm = jev(), case.llm()
    metric = case.metric(
        model=llm, system_one_model=model, eval_mode="llm", async_mode=is_async
    )
    score = await measure(metric, case.test_case(), is_async)
    assert not model.calls
    assert llm.was_sent(case.decision_key)
    assert score == pytest.approx(case.llm_score)


###############################################
# Metric-specific behaviour
###############################################

TRACE_METRICS = [StepEfficiencyMetric, PlanAdherenceMetric, PlanQualityMetric]


@pytest.mark.parametrize("cls", TRACE_METRICS, ids=lambda c: c.__name__)
@sync_and_async
@pytest.mark.asyncio
async def test_system_one_without_trace_sends_the_test_case(cls, is_async):
    model = jev()
    metric = cls(
        model=ExplodingLLM(),
        system_one_model=model,
        eval_mode="system_one",
        async_mode=is_async,
    )
    await measure(metric, traced_tc(trace=False), is_async)
    state = model.calls[0][0]
    assert "trace" not in state and "test_case" in state


@pytest.mark.parametrize(
    "cls", [StepEfficiencyMetric, PlanAdherenceMetric], ids=lambda c: c.__name__
)
@sync_and_async
@pytest.mark.asyncio
async def test_hybrid_without_trace_uses_the_llm(cls, is_async):
    model = jev()
    metric = cls(
        model=CannedLLM(PLAN),
        system_one_model=model,
        eval_mode="hybrid",
        async_mode=is_async,
    )
    score = await measure(metric, traced_tc(trace=False), is_async)
    assert score == pytest.approx(0.4)
    assert not model.calls


@pytest.mark.parametrize(
    "cls", [PlanAdherenceMetric, PlanQualityMetric], ids=lambda c: c.__name__
)
@sync_and_async
@pytest.mark.asyncio
async def test_system_one_no_plan_scores_one(cls, is_async):
    def no_plan(questions):
        out = answer_everything()(questions)
        for key in out.choices:
            out.choices[key] = ChoiceAnswer(
                choice="The agent states no plan",
                probabilities={"The agent states no plan": 0.95},
                confidence=0.9,
            )
        return out

    metric = cls(
        model=ExplodingLLM(),
        system_one_model=FakeSystemOneModel(answer_fn=no_plan),
        eval_mode="system_one",
        async_mode=is_async,
    )
    assert await measure(metric, traced_tc(), is_async) == 1.0


@sync_and_async
@pytest.mark.asyncio
async def test_tool_correctness_system_one_reports_tool_selection(is_async):
    metric = ToolCorrectnessMetric(
        available_tools=TOOLS,
        model=ExplodingLLM(),
        system_one_model=jev(),
        eval_mode="system_one",
        async_mode=is_async,
    )
    assert metric.model is None
    assert await measure(metric, traced_tc(), is_async) == 1.0
    assert "Decided by fake-jev: tool selection 1.00" in metric.reason


@sync_and_async
@pytest.mark.asyncio
async def test_tool_correctness_without_available_tools_skips_jev(is_async):
    model = jev()
    metric = ToolCorrectnessMetric(
        model=ExplodingLLM(),
        system_one_model=model,
        eval_mode="system_one",
        async_mode=is_async,
    )
    assert await measure(metric, traced_tc(), is_async) == 1.0
    assert not model.calls


@sync_and_async
@pytest.mark.asyncio
async def test_tool_correctness_system_one_raises_when_jev_fails(is_async):
    metric = ToolCorrectnessMetric(
        available_tools=TOOLS,
        system_one_model=jev_down(),
        eval_mode="system_one",
        async_mode=is_async,
    )
    with pytest.raises(Exception, match="down"):
        await measure(metric, traced_tc(), is_async)


class NameSchema(BaseModel):
    name: str


JSON_BAD = LLMTestCase(input="x", actual_output='{"nope": 1}')
JSON_GOOD = LLMTestCase(input="x", actual_output='{"name": "a"}')


@sync_and_async
@pytest.mark.asyncio
async def test_json_correctness_system_one_needs_no_model(is_async):
    metric = JsonCorrectnessMetric(
        expected_schema=NameSchema,
        model=ExplodingLLM(),
        eval_mode="system_one",
        async_mode=is_async,
    )
    assert metric.model is None and metric.evaluation_model is None
    assert metric.system_one_model is None

    assert await measure(metric, JSON_BAD, is_async) == 0
    assert "validation error" in metric.reason
    bad_reason = metric.reason
    assert await measure(metric, JSON_GOOD, is_async) == 1
    assert metric.reason.startswith("The generated Json")
    await measure(metric, JSON_BAD, is_async)
    assert metric.reason == bad_reason


@pytest.mark.parametrize("mode", ["hybrid", "llm"])
@sync_and_async
@pytest.mark.asyncio
async def test_json_correctness_llm_writes_reason_without_jev(mode, is_async):
    metric = JsonCorrectnessMetric(
        expected_schema=NameSchema,
        model=CannedLLM(json.dumps({"reason": "llm json reason"})),
        eval_mode=mode,
        async_mode=is_async,
    )
    assert metric.evaluation_model == "canned-llm"
    assert metric.system_one_model is None
    assert await measure(metric, JSON_BAD, is_async) == 0
    assert metric.reason == "llm json reason"
