from deepeval.telemetry import capture_tracing_integration
from deepeval.metrics import BaseMetric
from typing import List
import functools
import inspect
import json
from deepeval.test_case import LLMTestCase
from deepeval.tracing.types import TestCaseMetricPair
from deepeval.tracing.tracing import trace_manager

try:
    from opentelemetry.trace import NoOpTracer
    opentelemetry_installed = True
except:
    opentelemetry_installed = False

def is_opentelemetry_available():
    if not opentelemetry_installed:
        raise ImportError(
            "OpenTelemetry SDK is not available. Please install it with `pip install opentelemetry-sdk`."
        )
    return True

try:
    from pydantic_ai.agent import Agent
    from pydantic_ai.models.instrumented import InstrumentedModel

    pydantic_ai_installed = True
except:
    pydantic_ai_installed = False


def is_pydantic_ai_installed():
    if not pydantic_ai_installed:
        raise ImportError(
            "Pydantic AI is not installed. Please install it with `pip install pydantic-ai`."
        )


class PydanticAIAgent(Agent):
    def __init__(
        self,
        *args,
        metric_collection: str = None,
        metrics: List[BaseMetric] = None,
        trace_attributes: dict = None,
        **kwargs
    ):
        with capture_tracing_integration("pydantic_ai.agent.PydanticAIAgent"):
            is_pydantic_ai_installed()
            is_opentelemetry_available()
            
            super().__init__(*args, **kwargs)
            
            self.metric_collection = metric_collection
            self.trace_attributes = trace_attributes
            self.metrics = metrics
            
            # Patch the run method only for this instance
            self._patch_run_method()
    
    def _patch_run_method(self):
        """Patch the Agent.run method only for this PydanticAIAgent instance"""
        original_run = self.run
        
        @functools.wraps(original_run)
        async def patched_run(*args, **kwargs):
            model = kwargs.get("model", None)
            infer_name = kwargs.get("infer_name", True)

            if infer_name and self.name is None:
                self._infer_name(inspect.currentframe())
            model_used = self._get_model(model)
            del model
            
            if isinstance(model_used, InstrumentedModel):
                tracer = model_used.instrumentation_settings.tracer
            else:
                tracer = NoOpTracer()

            with tracer.start_as_current_span("agent") as run_span:
                result = await original_run(*args, **kwargs)

                name = "agent"
                if self.name:
                    name = str(self.name)
                
                input = ""
                if isinstance(args[0], str):
                    input = args[0]
                elif isinstance(args[0], list) and all(
                    isinstance(i, str) for i in args[0]
                ):
                    input = args[0]

                output = ""
                try:
                    output = str(result.output)
                except Exception:
                    pass
                
                # set agent span attributes
                run_span.set_attribute("confident.span.type", "agent")
                run_span.set_attribute("confident.agent.name", name)
                run_span.set_attribute("confident.agent.input", input)
                run_span.set_attribute("confident.agent.output", output)

                # fallback for input and output not being set
                run_span.set_attribute("confident.span.input", input)
                run_span.set_attribute("confident.span.output", output)

                # set metric collection attribute
                if self.metric_collection:
                    run_span.set_attribute("confident.span.metric_collection", self.metric_collection)
                
                # set trace attributes
                if self.trace_attributes:
                    if isinstance(self.trace_attributes, dict):
                        if self.trace_attributes.get("name"):
                            run_span.set_attribute("confident.trace.name", self.trace_attributes.get("name"))
                        if self.trace_attributes.get("tags"):
                            run_span.set_attribute("confident.trace.tags", self.trace_attributes.get("tags"))
                        if self.trace_attributes.get("metadata"):
                            run_span.set_attribute("confident.trace.metadata", json.dumps(self.trace_attributes.get("metadata")))
                        if self.trace_attributes.get("thread_id"):
                            run_span.set_attribute("confident.trace.thread_id", self.trace_attributes.get("thread_id"))
                        if self.trace_attributes.get("user_id"):
                            run_span.set_attribute("confident.trace.user_id", self.trace_attributes.get("user_id"))
                
                # set metrics
                if self.metrics:
                    trace_manager.test_case_metrics.append(
                        TestCaseMetricPair(
                            test_case=LLMTestCase(
                                input=input, actual_output=output
                            ),
                            metrics=self.metrics,
                        )
                    )
                

            return result
        
        # Replace the method only for this instance
        self.run = patched_run
