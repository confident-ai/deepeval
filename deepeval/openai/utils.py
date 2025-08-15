from typing import List
import uuid

from deepeval.tracing.types import ToolSpan, TraceSpanStatus
from deepeval.openai.extractors import InputParameters, OutputParameters
from deepeval.tracing.context import current_span_context
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.tracing.types import TestCaseMetricPair

openai_test_case_pairs: List[TestCaseMetricPair] = []


def set_attr_path(obj, attr_path: str, value):
    *pre_path, final_attr = attr_path.split(".")
    for attr in pre_path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return
    setattr(obj, final_attr, value)


def get_attr_path(obj, attr_path: str):
    for attr in attr_path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def add_test_case(
    test_case: LLMTestCase,
    metrics: List[BaseMetric],
    input_parameters: InputParameters,
):
    openai_test_case_pairs.append(
        TestCaseMetricPair(
            test_case=test_case,
            metrics=metrics,
            hyperparameters=create_hyperparameters_map(input_parameters),
        )
    )


def create_hyperparameters_map(input_parameters: InputParameters):
    hyperparameters = {"model": input_parameters.model}
    if input_parameters.instructions:
        hyperparameters["system_prompt"] = input_parameters.instructions
    elif input_parameters.messages:
        system_messages = [
            m["content"]
            for m in input_parameters.messages
            if m["role"] == "system"
        ]
        if system_messages:
            hyperparameters["system_prompt"] = (
                system_messages[0]
                if len(system_messages) == 1
                else str(system_messages)
            )
    return hyperparameters


def create_child_tool_spans(output_parameters: OutputParameters):
    if output_parameters.tools_called is None:
        return

    current_span = current_span_context.get()
    for tool_called in output_parameters.tools_called:
        tool_span = ToolSpan(
            **{
                "uuid": str(uuid.uuid4()),
                "trace_uuid": current_span.trace_uuid,
                "parent_uuid": current_span.uuid,
                "start_time": current_span.start_time,
                "end_time": current_span.start_time,
                "status": TraceSpanStatus.SUCCESS,
                "children": [],
                "name": tool_called.name,
                "input": tool_called.input_parameters,
                "output": None,
                "metrics": None,
                "description": tool_called.description,
            }
        )
        current_span.children.append(tool_span)
