from __future__ import annotations

import inspect
from typing import Any

import deepeval.metrics as metrics_pkg

from deepeval.metrics import BaseArenaMetric, BaseConversationalMetric
from deepeval.test_case import ConversationalTestCase, Turn, ToolCall

from tests.test_metrics.survey_mcp_fixtures import survey_mcp

_EXCLUDED = frozenset(
    {
        "ConversationalGEval",
        "ConversationalDAGMetric",
        "BaseConversationalMetric",
        "BaseMetric",
        "BaseArenaMetric",
    }
)

_TURN_TOOLS = [ToolCall(name="CheckDiscount"), ToolCall(name="CheckCars")]

_INIT_DEFAULTS: dict[str, Any] = {
    "relevant_topics": ["refunds", "shipping", "policies"],
    "available_tools": _TURN_TOOLS,
}


def conversational_metrics() -> list[tuple[str, type[BaseConversationalMetric]]]:
    out: list[tuple[str, type[BaseConversationalMetric]]] = []
    for name in metrics_pkg.__all__:
        if name in _EXCLUDED:
            continue
        cls = getattr(metrics_pkg, name, None)
        if not isinstance(cls, type):
            continue
        if issubclass(cls, BaseArenaMetric):
            continue
        if issubclass(cls, BaseConversationalMetric):
            out.append((name, cls))
    return sorted(out)


def make_conversational_metric(
    cls: type[BaseConversationalMetric],
) -> BaseConversationalMetric | None:
    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}
    for pname, param in list(sig.parameters.items())[1:]:
        if param.default is inspect.Parameter.empty:
            if pname not in _INIT_DEFAULTS:
                return None
            kwargs[pname] = _INIT_DEFAULTS[pname]
    if "async_mode" in sig.parameters:
        kwargs["async_mode"] = False
    try:
        return cls(**kwargs)
    except Exception:
        return None


def conversational_test_case() -> ConversationalTestCase:
    mcp = survey_mcp()
    return ConversationalTestCase(
        scenario="Testing scenario for survey.",
        context=["Testing context line one.", "Testing context line two."],
        name="survey-conversational-case",
        user_description="Testing shopper persona.",
        expected_outcome="The chatbot must explain store policies including refunds.",
        chatbot_role="A helpful assistant",
        metadata={"survey": "Testing"},
        comments="Testing",
        tags=["survey", "testing"],
        mcp_servers=[mcp["server"]],
        turns=[
            Turn(
                role="user",
                content="What if these shoes don't fit?",
                user_id="survey-user-1",
                metadata={"turn": "Testing"},
                tools_called=_TURN_TOOLS,
                mcp_tools_called=mcp["tools"],
            ),
            Turn(
                role="assistant",
                content="We offer a 30-day full refund at no extra cost.",
                user_id="survey-assistant-1",
                retrieval_context=[
                    "All customers are eligible for a 30 day full refund at no extra cost."
                ],
                tools_called=_TURN_TOOLS,
                mcp_tools_called=mcp["tools"],
                metadata={"turn": "Testing"},
            ),
        ],
    )
