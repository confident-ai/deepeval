from __future__ import annotations

import inspect
from typing import Any

import deepeval.metrics as metrics_pkg
from pydantic import BaseModel

from deepeval.metrics import BaseArenaMetric, BaseConversationalMetric, BaseMetric
from deepeval.test_case import LLMTestCase, MLLMImage, ToolCall

from tests.test_metrics.survey_mcp_fixtures import survey_mcp

_EXCLUDED = frozenset(
    {
        "GEval",
        "ArenaGEval",
        "ConversationalGEval",
        "DAGMetric",
        "ConversationalDAGMetric",
        "DeepAcyclicGraph",
        "BaseMetric",
        "BaseConversationalMetric",
        "BaseArenaMetric",
        "TaskCompletionMetric",
        "StepEfficiencyMetric",
        "PlanAdherenceMetric",
        "PlanQualityMetric",
    }
)

_IMAGE_METRICS = frozenset(
    {
        "TextToImageMetric",
        "ImageEditingMetric",
        "ImageCoherenceMetric",
        "ImageHelpfulnessMetric",
        "ImageReferenceMetric",
    }
)

_TOOLS = [ToolCall(name="ImageAnalysis"), ToolCall(name="ToolQuery")]


class SurveyJsonModel(BaseModel):
    summary: str = "ok"


_INIT_DEFAULTS: dict[str, Any] = {
    "pattern": r"^[\s\S]{1,5000}$",
    "expected_schema": SurveyJsonModel,
    "domain": "financial",
    "advice_types": ["medical"],
    "prompt_instructions": "Answer briefly and follow the user request.",
}


def single_turn_metrics() -> list[tuple[str, type[BaseMetric]]]:
    out: list[tuple[str, type[BaseMetric]]] = []
    for name in metrics_pkg.__all__:
        if name in _EXCLUDED:
            continue
        cls = getattr(metrics_pkg, name, None)
        if not isinstance(cls, type):
            continue
        if issubclass(cls, (BaseArenaMetric, BaseConversationalMetric)):
            continue
        if issubclass(cls, BaseMetric):
            out.append((name, cls))
    return sorted(out)


def make_single_turn_metric(cls: type[BaseMetric]) -> BaseMetric | None:
    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}
    for pname, param in list(sig.parameters.items())[1:]:
        if param.default is inspect.Parameter.empty:
            if pname not in _INIT_DEFAULTS:
                return None
            kwargs[pname] = _INIT_DEFAULTS[pname]
    if "role" in sig.parameters:
        kwargs.setdefault("role", "helpful assistant")
    if "async_mode" in sig.parameters:
        kwargs["async_mode"] = False
    try:
        return cls(**kwargs)
    except Exception:
        return None


def llm_test_case(metric_name: str, *, car_path: str) -> LLMTestCase:
    mcp = survey_mcp()
    common = dict(
        name="survey-llm-case",
        tags=["survey", "testing"],
        comments="Testing",
        metadata={"survey": "Testing"},
        token_cost=0.01,
        completion_time=0.5,
        custom_column_key_values={"survey_col": "Testing"},
        mcp_servers=[mcp["server"]],
        mcp_tools_called=mcp["tools"],
        tools_called=_TOOLS,
        expected_tools=[ToolCall(name="ImageAnalysis")],
    )
    if metric_name in _IMAGE_METRICS:
        image = MLLMImage(url=car_path)
        image_common = dict(
            retrieval_context=[f"Cars are great to look at {image}"],
            context=[f"Cars are great to look at {image}"],
            **common,
        )
        if metric_name == "TextToImageMetric":
            return LLMTestCase(
                input="Generate an image of a car.",
                expected_output=f"That's an image of a car {image}",
                actual_output=f"That's an image of a car {image}",
                **image_common,
            )
        if metric_name == "ImageEditingMetric":
            return LLMTestCase(
                input=f"Edit this image to make it blue. {image}",
                expected_output=f"That's an image of a car {image}",
                actual_output=f"That is a car {image}.",
                **image_common,
            )
        return LLMTestCase(
            input=f"What's shown in this image? {image}",
            expected_output="That's an image of a car",
            actual_output=f"That is a car {image}.",
            **image_common,
        )
    policy = "We offer a 30-day full refund at no extra cost."
    context = ["All customers are eligible for a 30 day full refund at no extra cost."]
    return LLMTestCase(
        input="What is the return policy for shoes?",
        expected_output=policy,
        actual_output=(
            SurveyJsonModel().model_dump_json()
            if metric_name == "JsonCorrectnessMetric"
            else policy
        ),
        retrieval_context=context,
        context=context,
        **common,
    )
