from __future__ import annotations

from dataclasses import replace

from agents import (
    Runner as BaseRunner,
    RunConfig,
    RunResult,
    RunResultStreaming,
)
from deepeval.tracing.tracing import Observer
from deepeval.tracing.context import current_span_context, current_trace_context

# Import observed provider/model helpers from our agent module
from deepeval.openai_agents.agent import _ObservedProvider


class Runner(BaseRunner):
    """
    Extends Runner to:
      - capture metric_collection/metrics at run entry for tracing
      - ensure RunConfig.model_provider is wrapped to return observed Models
        so string-based model lookups are also instrumented.
    """

    @classmethod
    async def run(cls, *args, **kwargs) -> RunResult:
        metric_collection = kwargs.pop("metric_collection", None)
        metrics = kwargs.pop("metrics", None)

        # Ensure the model provider is wrapped so _get_model(...) uses observed Models
        starting_agent = (
            args[0] if len(args) > 0 else kwargs.get("starting_agent")
        )
        run_config: RunConfig | None = kwargs.get("run_config")
        if run_config is None:
            run_config = RunConfig()
            kwargs["run_config"] = run_config

        if run_config.model_provider is not None:
            run_config.model_provider = _ObservedProvider(
                run_config.model_provider,
                metrics=getattr(starting_agent, "metrics", None) or metrics,
                metric_collection=getattr(
                    starting_agent, "metric_collection", None
                )
                or metric_collection,
                deepeval_prompt=getattr(
                    starting_agent, "deepeval_prompt", None
                ),
            )

        input_val = args[1] if len(args) >= 2 else kwargs.get("input", None)
        with Observer(
            span_type="custom",
            metric_collection=metric_collection,
            metrics=metrics,
            func_name="run",
            function_kwargs={"input": input_val},
        ) as observer:
            current_span = current_span_context.get()
            current_trace = current_trace_context.get()
            current_trace.input = input_val
            if current_span:
                current_span.input = input_val
            res = await super().run(*args, **kwargs)
            current_trace.output = str(res)
            observer.result = str(res)
        return res

    @classmethod
    def run_sync(cls, *args, **kwargs) -> RunResult:
        metric_collection = kwargs.pop("metric_collection", None)
        metrics = kwargs.pop("metrics", None)

        starting_agent = (
            args[0] if len(args) > 0 else kwargs.get("starting_agent")
        )
        run_config: RunConfig | None = kwargs.get("run_config")
        if run_config is None:
            run_config = RunConfig()
            kwargs["run_config"] = run_config

        if run_config.model_provider is not None:
            run_config.model_provider = _ObservedProvider(
                run_config.model_provider,
                metrics=getattr(starting_agent, "metrics", None) or metrics,
                metric_collection=getattr(
                    starting_agent, "metric_collection", None
                )
                or metric_collection,
                deepeval_prompt=getattr(
                    starting_agent, "deepeval_prompt", None
                ),
            )

        input_val = args[1] if len(args) >= 2 else kwargs.get("input", None)
        with Observer(
            span_type="custom",
            metric_collection=metric_collection,
            metrics=metrics,
            func_name="run_sync",
            function_kwargs={"input": input_val},
        ) as observer:
            current_span = current_span_context.get()
            current_trace = current_trace_context.get()
            current_trace.input = input_val
            if current_span:
                current_span.input = input_val
            res = super().run_sync(*args, **kwargs)
            current_trace.output = str(res)
            observer.result = str(res)

        return res
