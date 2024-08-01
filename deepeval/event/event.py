from typing import Optional, List, Dict, Union, Any

from deepeval.monitor.monitor import monitor
from deepeval.monitor.api import Link


def track(
    event_name: str,
    model: str,
    input: str,
    response: str,
    retrieval_context: Optional[List[str]] = None,
    completion_time: Optional[float] = None,
    token_usage: Optional[float] = None,
    token_cost: Optional[float] = None,
    distinct_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    additional_data: Optional[
        Dict[str, Union[str, Link, List[Link], Dict]]
    ] = None,
    hyperparameters: Optional[Dict[str, str]] = {},
    fail_silently: Optional[bool] = False,
    raise_expection: Optional[bool] = True,
    run_async: Optional[bool] = True,
    trace_stack: Optional[Dict[str, Any]] = None,
    trace_provider: Optional[str] = None,
) -> str:
    print(
        "deepeval.track(...) will be deprecated soon. Please switch over to deepeval.monitor(...)"
    )
    return monitor(
        event_name=event_name,
        model=model,
        input=input,
        response=response,
        retrieval_context=retrieval_context,
        completion_time=completion_time,
        token_cost=token_cost,
        token_usage=token_usage,
        distinct_id=distinct_id,
        conversation_id=conversation_id,
        additional_data=additional_data,
        hyperparameters=hyperparameters,
        fail_silently=fail_silently,
        raise_expection=raise_expection,
        run_async=run_async,
        trace_stack=trace_stack,
        trace_provider=trace_provider,
    )
