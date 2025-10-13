from deepeval.tracing.otel.exporter import ConfidentSpanExporter
from tests.test_integrations.test_exporter.readable_spans import (
    list_of_readable_spans,
)

exporter = ConfidentSpanExporter()

from deepeval.tracing.trace_test_manager import trace_testing_manager


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
