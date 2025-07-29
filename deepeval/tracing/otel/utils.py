import json
from typing import List
from deepeval.tracing.types import Trace
from deepeval.tracing.types import LLMTestCase
from deepeval.test_case import ToolCall

def to_hex_string(id_value: int | bytes, length: int = 32) -> str:
    """
    Convert a trace ID or span ID to a hex string.

    Args:
        id_value: The ID value to convert, either as an integer or bytes
        length: The expected length of the hex string (32 for trace IDs, 16 for span IDs)

    Returns:
        A hex string representation of the ID
    """
    if isinstance(id_value, int):
        return format(id_value, f"0{length}x")
    return id_value.hex()

def set_trace_time(trace: Trace):
    """
    Set the trace time based on the root span with the largest start and end time gap.
    
    Args:
        trace: The trace object to update
    """

    if not trace.root_spans:
        return
    
    # Find the root span with the largest time gap
    max_gap = 0
    target_span = None
    
    for span in trace.root_spans:
        # Skip spans that don't have both start and end times
        if span.end_time is None:
            continue
            
        # Calculate the time gap
        time_gap = span.end_time - span.start_time
        
        # Update if this span has a larger gap
        if time_gap > max_gap:
            max_gap = time_gap
            target_span = span
    
    # If we found a valid span, set the trace time to match
    if target_span is not None:
        trace.start_time = target_span.start_time
        trace.end_time = target_span.end_time

def validate_llm_test_case_data(input, actual_output, expected_output, context, retrieval_context, tools_called, expected_tools):
    """Validate LLMTestCase data before creation"""
    if not isinstance(input, str):
        raise ValueError(f"input must be a string, got {type(input)}")
    
    if not isinstance(actual_output, str):
        raise ValueError(f"actual_output must be a string, got {type(actual_output)}")
    
    if expected_output is not None and not isinstance(expected_output, str):
        raise ValueError(f"expected_output must be None or a string, got {type(expected_output)}")
    
    if context is not None:
        if not isinstance(context, list) or not all(isinstance(item, str) for item in context):
            raise ValueError("context must be None or a list of strings")
    
    if retrieval_context is not None:
        if not isinstance(retrieval_context, list) or not all(isinstance(item, str) for item in retrieval_context):
            raise ValueError("retrieval_context must be None or a list of strings")
    
    if tools_called is not None:
        if not isinstance(tools_called, list) or not all(isinstance(item, ToolCall) for item in tools_called):
            raise ValueError("tools_called must be None or a list of ToolCall objects")
    
    if expected_tools is not None:
        if not isinstance(expected_tools, list) or not all(isinstance(item, ToolCall) for item in expected_tools):
            raise ValueError("expected_tools must be None or a list of ToolCall objects")
