import asyncio
from deepeval.tracing.otel.exporter import ConfidentSpanExporter
from tests.test_integrations.test_exporter.readable_spans import (
    list_of_readable_spans,
    llm_span_list,
)
from deepeval.tracing.trace_test_manager import trace_testing_manager


exporter = ConfidentSpanExporter()


async def test_pydantic_ai_trace():
    try:
        trace_testing_manager.test_name = "any_name"
        exporter.export(list_of_readable_spans)
        actual_dict = await trace_testing_manager.wait_for_test_dict()

        # Assert that System Instruction is the first input message
        assert (
            actual_dict["input"][0]["role"] == "System Instruction"
        ), f"Expected first input role to be 'System Instruction', got {actual_dict['input'][0]['role']}"

        # Assert that output is the last non-thinking part (the final text content)
        assert (
            actual_dict["output"]["content"] == "Final response text"
        ), f"Expected output content to be 'Final response text', got {actual_dict['output']['content']}"

    finally:
        trace_testing_manager.test_name = None
        trace_testing_manager.test_dict = None


async def test_llm_trace():
    try:
        trace_testing_manager.test_name = "any_name"
        exporter.export(llm_span_list)
        actual_dict = await trace_testing_manager.wait_for_test_dict()

        assert (
            actual_dict["llmSpans"][0]["input"][-1]["role"]
            == "Model Request Parameters"
        ), f"Expected input role to be 'Model Request Parameters', got {actual_dict['llmSpans'][0]['input'][-1]['role']}"

        assert (
            actual_dict["llmSpans"][0]["inputTokenCount"] == 1000
        ), f"Expected input token count to be 1000, got {actual_dict['llmSpans'][0]['inputTokenCount']}"
        assert (
            actual_dict["llmSpans"][0]["outputTokenCount"] == 500
        ), f"Expected output token count to be 500, got {actual_dict['llmSpans'][0]['outputTokenCount']}"

    finally:
        trace_testing_manager.test_name = None
        trace_testing_manager.test_dict = None
