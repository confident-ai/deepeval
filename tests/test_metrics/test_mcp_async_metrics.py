import json
import pytest
from mcp.types import CallToolResult, TextContent

from deepeval.metrics import MCPTaskCompletionMetric, MultiTurnMCPUseMetric
from deepeval.metrics.mcp.schema import Reason
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    ConversationalTestCase,
    MCPServer,
    MCPToolCall,
    Turn,
)


class _FakeMCPLLM(DeepEvalBaseLLM):
    def __init__(self, *, allow_sync: bool = False, step_score: float = 1.0):
        self.allow_sync = allow_sync
        self.step_score = step_score
        self.sync_calls = 0
        self.async_calls = 0
        self.reason_prompts = []
        super().__init__(model="fake-mcp-judge")

    def load_model(self):
        return self

    def _reply(self, prompt: str, schema=None) -> str:
        if schema is Reason:
            self.reason_prompts.append(prompt)
            return json.dumps({"reason": "synthetic final reason"})
        if schema is not None:
            fields = getattr(schema, "model_fields", {})
            if "tasks" in fields:
                return json.dumps(
                    {
                        "tasks": [
                            {
                                "task": "Check weather in Paris.",
                                "steps_taken": [
                                    "Called get_weather(city=Paris)"
                                ],
                            }
                        ]
                    }
                )
            return json.dumps(
                {
                    "score": self.step_score,
                    "reason": "synthetic step reason",
                }
            )
        return json.dumps({"tasks": []})

    def generate(self, prompt: str, schema=None):
        self.sync_calls += 1
        if not self.allow_sync:
            raise AssertionError(
                "sync generate() must not be called in a_measure()"
            )
        return self._reply(prompt, schema=schema)

    async def a_generate(self, prompt: str, schema=None):
        self.async_calls += 1
        return self._reply(prompt, schema=schema)

    def get_model_name(self) -> str:
        return "fake-mcp-judge"


def _mcp_convo() -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Check weather in Paris via MCP."),
            Turn(
                role="assistant",
                content="Calling weather tool.",
                mcp_tools_called=[
                    MCPToolCall(
                        name="get_weather",
                        args={"city": "Paris"},
                        result=CallToolResult(
                            content=[TextContent(type="text", text="sunny")]
                        ),
                    )
                ],
            ),
            Turn(role="assistant", content="It is sunny in Paris."),
        ],
        expected_outcome="Report the Paris weather using the MCP tool.",
        mcp_servers=[MCPServer(server_name="weather")],
    )


@pytest.mark.asyncio
async def test_mcp_task_completion_a_measure_uses_async_reason_and_current_success():
    model = _FakeMCPLLM(allow_sync=False, step_score=1.0)
    metric = MCPTaskCompletionMetric(model=model, include_reason=True)

    score = await metric.a_measure(_mcp_convo(), _show_indicator=False)
    assert score == 1.0
    assert metric.reason == "synthetic final reason"
    assert model.sync_calls == 0
    assert model.async_calls >= 2
    assert "Success: True" in model.reason_prompts[-1]

    model.step_score = 0.0
    score2 = await metric.a_measure(_mcp_convo(), _show_indicator=False)
    assert score2 == 0.0
    assert model.sync_calls == 0
    assert "Success: False" in model.reason_prompts[-1]


@pytest.mark.asyncio
async def test_multi_turn_mcp_use_a_measure_uses_async_reason_and_current_success():
    model = _FakeMCPLLM(allow_sync=False, step_score=1.0)
    metric = MultiTurnMCPUseMetric(model=model, include_reason=True)

    score = await metric.a_measure(_mcp_convo(), _show_indicator=False)
    assert score == 1.0
    assert metric.reason == "synthetic final reason"
    assert model.sync_calls == 0
    assert model.async_calls >= 3
    assert "Success: True" in model.reason_prompts[-1]

    model.step_score = 0.0
    score2 = await metric.a_measure(_mcp_convo(), _show_indicator=False)
    assert score2 == 0.0
    assert model.sync_calls == 0
    assert "Success: False" in model.reason_prompts[-1]


def test_mcp_metrics_sync_measure_computes_success_before_reason():
    for metric_cls in (MCPTaskCompletionMetric, MultiTurnMCPUseMetric):
        model = _FakeMCPLLM(allow_sync=True, step_score=0.0)
        metric = metric_cls(model=model, include_reason=True, async_mode=False)

        score = metric.measure(_mcp_convo(), _show_indicator=False)
        assert score == 0.0
        assert metric.success is False
        assert "Success: False" in model.reason_prompts[-1]
