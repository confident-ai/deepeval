import json
from typing import List
from deepeval.tracing.types import Trace
from deepeval.tracing.types import LLMTestCase

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


def validate_and_prepare_llm_test_cases(llm_test_case_json_list: List[str]):
    """
    Validate and prepare the LLM test case.
    
    Args:
        llm_test_case: The LLM test case to validate and prepare
    """
    llm_test_cases = []

    if not isinstance(llm_test_case_json_list, list):
        return llm_test_cases
    
    for llm_test_case_str in llm_test_case_json_list:
        try:
            llm_test_case_json = json.loads(llm_test_case_str)
            llm_test_case = LLMTestCase(
                input=llm_test_case_json.get('input'),
                actual_output=llm_test_case_json.get('actual_output'),
                context=llm_test_case_json.get('context'),
                retrieval_context=llm_test_case_json.get('retrieval_context'),
                additional_metadata=llm_test_case_json.get('additional_metadata'),
                tools_called=llm_test_case_json.get('tools_called'),
                comments=llm_test_case_json.get('comments'),
                expected_tools=llm_test_case_json.get('expected_tools'),
                token_cost=llm_test_case_json.get('token_cost'),
                completion_time=llm_test_case_json.get('completion_time'),
                name=llm_test_case_json.get('name'),
                tags=llm_test_case_json.get('tags'),
                _trace_dict=llm_test_case_json.get('_trace_dict'),
                _dataset_rank=llm_test_case_json.get('_dataset_rank'),
                _dataset_alias=llm_test_case_json.get('_dataset_alias'),
                _dataset_id=llm_test_case_json.get('_dataset_id'),
                _identifier=llm_test_case_json.get('_identifier'),
            )
            llm_test_cases.append(llm_test_case)
        except Exception as e:
            print(f"Error validating LLM test case: {e}")
            continue

    return llm_test_cases
