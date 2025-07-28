from deepeval.tracing.types import Trace
import time
from datetime import datetime, timezone

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
    
def convert_to_perf_counter(timestamp: int) -> float:
    timestamp_seconds = timestamp / 1e9
    current_time = datetime.now(timezone.utc)
    current_perf_counter = time.perf_counter()

    offset = current_perf_counter - current_time.timestamp()
    return timestamp_seconds + offset