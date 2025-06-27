import functools
from typing import Any, Optional
from deepeval.tracing import trace_manager
from deepeval.tracing.types import LlmSpan, LlmAttributes, TraceSpanStatus
from uuid import uuid4
from time import perf_counter

try:
    from crewai import LLM

    crewai_installed = True
except:
    crewai_installed = False


def is_crewai_installed():
    if not crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )


class CrewAILogger:
    active_trace_id: Optional[str] = None

    def __init__(self):
        is_crewai_installed()

    def patch_crewai_LLM(self, method_to_patch: str):
        original_methods = {}

        method = getattr(LLM, method_to_patch)
        if callable(method) and not isinstance(method, type):
            original_methods[method_to_patch] = method

            @functools.wraps(method)
            def wrapped_method(*args, original_method=method, **kwargs):
                if self.active_trace_id is None:
                    self.active_trace_id = trace_manager.start_new_trace().uuid

                llm_span = LlmSpan(
                    uuid=str(uuid4()),
                    status=TraceSpanStatus.IN_PROGRESS,
                    children=[],
                    trace_uuid=self.active_trace_id,
                    parent_uuid=None,
                    start_time=perf_counter(),
                    name="crewai_llm_span_" + str(uuid4()),
                    # TODO: why model is coming unknown?
                    model="unknown",
                    attributes=LlmAttributes(input=args[1], output=""),
                )
                trace_manager.add_span(llm_span)
                trace_manager.add_span_to_trace(llm_span)

                response = original_method(*args, **kwargs)

                llm_span.end_time = perf_counter()
                llm_span.status = TraceSpanStatus.SUCCESS
                llm_span.set_attributes(
                    LlmAttributes(
                        input=llm_span.attributes.input, output=response
                    )
                )
                trace_manager.remove_span(llm_span.uuid)
                trace_manager.end_trace(self.active_trace_id)
                self.active_trace_id = None

                return response

            setattr(LLM, method_to_patch, wrapped_method)
