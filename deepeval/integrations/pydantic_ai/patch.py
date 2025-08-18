import json
import inspect
from contextlib import asynccontextmanager
from deepeval.integrations.pydantic_ai import Agent as PatchedAgent
from opentelemetry import trace
from opentelemetry.trace import NoOpTracer
from deepeval.tracing.tracing import trace_manager
from deepeval.test_case import LLMTestCase
from deepeval.tracing.types import TestCaseMetricPair

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.instrumented import InstrumentedModel

    pydantic_ai_installed = True
except:
    pydantic_ai_installed = False


def is_pydantic_ai_installed():
    if not pydantic_ai_installed:
        raise ImportError(
            "Pydantic AI is not installed. Please install it with `pip install pydantic-ai`."
        )


def safe_patch_agent_run_method():
    is_pydantic_ai_installed()
    original_run = Agent.run

    # define patched run method
    async def patched_run(*args, **kwargs):

        model = kwargs.get("model", None)
        infer_name = kwargs.get("infer_name", True)

        if infer_name and args[0].name is None:
            args[0]._infer_name(inspect.currentframe())
        model_used = args[0]._get_model(model)
        del model

        if isinstance(model_used, InstrumentedModel):
            tracer = model_used.instrumentation_settings.tracer
        else:
            tracer = NoOpTracer()

        with tracer.start_as_current_span("agent") as run_span:
            result = await original_run(*args, **kwargs)

            name = "agent"
            if isinstance(args[0], PatchedAgent):
                name = str(args[0].name)

            input = ""
            if isinstance(args[1], str):
                input = args[1]
            elif isinstance(args[1], list) and all(
                isinstance(i, str) for i in args[1]
            ):
                input = args[1]

            output = ""

            # Check if result.output is convertible to string
            try:
                output = str(result.output)
            except Exception:
                pass

            # agent attributes
            run_span.set_attribute("confident.span.type", "agent")
            run_span.set_attribute("confident.agent.name", name)
            run_span.set_attribute("confident.agent.input", input)
            run_span.set_attribute("confident.agent.output", output)

            # llm test case attributes
            if isinstance(args[0], PatchedAgent):
                if args[0].metric_collection:
                    run_span.set_attribute(
                        "confident.span.metric_collection",
                        args[0].metric_collection,
                    )

                if args[0].trace_attributes:
                    if isinstance(args[0].trace_attributes, dict):
                        if args[0].trace_attributes.get("name"):
                            run_span.set_attribute(
                                "confident.trace.name",
                                args[0].trace_attributes.get("name"),
                            )
                        if args[0].trace_attributes.get("tags"):
                            run_span.set_attribute(
                                "confident.trace.tags",
                                args[0].trace_attributes.get("tags"),
                            )
                        if args[0].trace_attributes.get("thread_id"):
                            run_span.set_attribute(
                                "confident.trace.thread_id",
                                args[0].trace_attributes.get("thread_id"),
                            )
                        if args[0].trace_attributes.get("user_id"):
                            run_span.set_attribute(
                                "confident.trace.user_id",
                                args[0].trace_attributes.get("user_id"),
                            )
                        if args[0].trace_attributes.get("metadata"):
                            run_span.set_attribute(
                                "confident.trace.metadata",
                                json.dumps(
                                    args[0].trace_attributes.get("metadata")
                                ),
                            )
                        if args[0].trace_attributes.get("environment"):
                            run_span.set_attribute(
                                "confident.trace.environment",
                                args[0].trace_attributes.get("environment"),
                            )

            run_span.set_attribute("confident.span.input", input)
            run_span.set_attribute("confident.span.output", output)

            if args[0].metrics:
                trace_manager.test_case_metrics.append(
                    TestCaseMetricPair(
                        test_case=LLMTestCase(
                            input=input, actual_output=output
                        ),
                        metrics=args[0].metrics,
                    )
                )

        return result

    # Apply the patch
    Agent.run = patched_run


def safe_patch_agent_iter_method():
    is_pydantic_ai_installed()
    original_iter = Agent.iter

    @asynccontextmanager
    async def patched_iter(self, *args, **kwargs):
        """
        A patched version of Agent.iter that captures the run_span
        and adds a custom attribute to it.
        """
        # Call the original iter method as an async context manager
        async with original_iter(self, *args, **kwargs) as agent_run:

            # Because we are inside the context of the original iter,
            # the run_span is active. We can get it.
            run_span = trace.get_current_span()

            # Now you can set any attributes you want on the span
            run_span.set_attribute("confident.span.name", "agent iter")

            # Yield the agent_run to the original caller
            yield agent_run

    Agent.iter = patched_iter
