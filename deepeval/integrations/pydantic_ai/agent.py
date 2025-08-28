from deepeval.telemetry import capture_tracing_integration
from deepeval.metrics import BaseMetric
from typing import List
import functools
import inspect
import json
from deepeval.test_case import LLMTestCase
from deepeval.tracing.types import TestCaseMetricPair
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.otel.utils import parse_string, parse_list_of_strings

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
    def __init__(self, *args, **kwargs):
        with capture_tracing_integration("pydantic_ai.agent.PydanticAIAgent"):
            is_pydantic_ai_installed()
            is_opentelemetry_available()

            super().__init__(*args, **kwargs)

            # attributes to be set if ran synchronously
            self.metric_collection: str = None
            self.metrics: list[BaseMetric] = None

            # trace attributes to be set if ran synchronously
            self._trace_name: str = None
            self._trace_tags: list[str] = None
            self._trace_metadata: dict = None
            self._trace_thread_id: str = None
            self._trace_user_id: str = None

            # Patch the run method only for this instance
            self._patch_run_method()
            self._patch_run_method_sync()

    def _patch_run_method(self):
        """Patch the Agent.run method only for this PydanticAIAgent instance"""
        original_run = self.run

        @functools.wraps(original_run)
        async def patched_run(
            *args,
            metric_collection=None,
            metrics=None,
            trace_name=None,
            trace_tags=None,
            trace_metadata=None,
            trace_thread_id=None,
            trace_user_id=None,
            **kwargs
        ):
            # extract and validate flattened arguments - use safe pop with defaults
            metric_collection = parse_string(metric_collection)
            trace_name = parse_string(trace_name)
            trace_tags = parse_list_of_strings(trace_tags)
            trace_thread_id = parse_string(trace_thread_id)
            trace_user_id = parse_string(trace_user_id)

            if metrics is not None and not (
                isinstance(metrics, list)
                and all(isinstance(m, BaseMetric) for m in metrics)
            ):
                raise TypeError(
                    "metrics must be a list of BaseMetric instances"
                )

            if trace_metadata is not None and not isinstance(
                trace_metadata, dict
            ):
                raise TypeError("trace_metadata must be a dictionary")

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

                if metric_collection:  # flattened argument to be replaced
                    run_span.set_attribute(
                        "confident.span.metric_collection", metric_collection
                    )
                elif self.metric_collection:  # for run_sync
                    run_span.set_attribute(
                        "confident.span.metric_collection",
                        self.metric_collection,
                    )

                # set the flattened trace attributes
                if trace_name:
                    run_span.set_attribute("confident.trace.name", trace_name)
                if trace_tags:
                    run_span.set_attribute("confident.trace.tags", trace_tags)
                if trace_metadata:
                    run_span.set_attribute(
                        "confident.trace.metadata", json.dumps(trace_metadata)
                    )
                if trace_thread_id:
                    run_span.set_attribute(
                        "confident.trace.thread_id", trace_thread_id
                    )
                if trace_user_id:
                    run_span.set_attribute(
                        "confident.trace.user_id", trace_user_id
                    )

                # for run_sync
                if self._trace_name:
                    run_span.set_attribute(
                        "confident.trace.name", self._trace_name
                    )
                if self._trace_tags:
                    run_span.set_attribute(
                        "confident.trace.tags", self._trace_tags
                    )
                if self._trace_metadata:
                    run_span.set_attribute(
                        "confident.trace.metadata",
                        json.dumps(self._trace_metadata),
                    )
                if self._trace_thread_id:
                    run_span.set_attribute(
                        "confident.trace.thread_id", self._trace_thread_id
                    )
                if self._trace_user_id:
                    run_span.set_attribute(
                        "confident.trace.user_id", self._trace_user_id
                    )

                if metrics:  # flattened argument to be replaced
                    trace_manager.test_case_metrics.append(
                        TestCaseMetricPair(
                            test_case=LLMTestCase(
                                input=input, actual_output=output
                            ),
                            metrics=metrics,
                        )
                    )
                elif self.metrics:  # for run_sync
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

    def _patch_run_method_sync(self):
        """Patch the Agent.run method only for this PydanticAIAgent instance"""
        original_run = self.run_sync

        @functools.wraps(original_run)
        def patched_run(
            *args,
            metric_collection=None,
            metrics=None,
            trace_name=None,
            trace_tags=None,
            trace_metadata=None,
            trace_thread_id=None,
            trace_user_id=None,
            **kwargs
        ):
            metric_collection = parse_string(metric_collection)
            trace_name = parse_string(trace_name)
            trace_tags = parse_list_of_strings(trace_tags)
            trace_thread_id = parse_string(trace_thread_id)
            trace_user_id = parse_string(trace_user_id)

            if metrics is not None and not (
                isinstance(metrics, list)
                and all(isinstance(m, BaseMetric) for m in metrics)
            ):
                raise TypeError(
                    "metrics must be a list of BaseMetric instances"
                )

            if trace_metadata is not None and not isinstance(
                trace_metadata, dict
            ):
                raise TypeError("trace_metadata must be a dictionary")

            # attributes to be set if ran synchronously
            if metric_collection:
                self.metric_collection = metric_collection
            if metrics:
                self.metrics = metrics

            self._trace_name = trace_name
            self._trace_tags = trace_tags
            self._trace_metadata = trace_metadata
            self._trace_thread_id = trace_thread_id
            self._trace_user_id = trace_user_id

            result = original_run(*args, **kwargs)

            return result

        # Replace the method only for this instance
        self.run_sync = patched_run
