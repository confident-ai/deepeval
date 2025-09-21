from __future__ import annotations

from dataclasses import replace
from typing import List

from agents import (
    RunConfig,
    RunResult,
    RunResultStreaming,
    Runner,
)
from agents.agent import Agent
from agents.items import TResponseInputItem
from agents.lifecycle import RunHooks
from agents.memory import Session
from agents.run import DEFAULT_MAX_TURNS
from agents.run_context import TContext
from deepeval.tracing.tracing import Observer
from deepeval.tracing.context import current_span_context, current_trace_context

# Import observed provider/model helpers from our agent module
from deepeval.openai_agents.agent import _ObservedProvider
from deepeval.metrics import BaseMetric


class Runner(Runner):
    """
    Extends Runner to capture metric_collection/metrics at run entry for tracing
    and ensure RunConfig.model_provider is wrapped to return observed Models
    so string-based model lookups are also instrumented.
    """

    @classmethod
    async def run(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        *,
        context: TContext | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        session: Session | None = None,

        metrics: List[BaseMetric] | None = None,
        metric_collection: str | None = None,
    ) -> RunResult:
        # Ensure the model provider is wrapped so _get_model(...) uses observed Models
        if run_config is None:
            run_config = RunConfig()

        if run_config.model_provider is not None:
            run_config.model_provider = _ObservedProvider(
                run_config.model_provider,
                metrics=getattr(starting_agent, "llm_metrics", None),
                metric_collection=getattr(starting_agent, "llm_metric_collection", None),
                deepeval_prompt=getattr(starting_agent, "deepeval_prompt", None),
            )

        with Observer(
            span_type="custom",
            metric_collection=metric_collection,
            metrics=metrics,
            func_name="run",
            function_kwargs={"input": input},
        ) as observer:
            current_span = current_span_context.get()
            current_trace = current_trace_context.get()
            current_trace.input = input
            if current_span:
                current_span.input = input
            res = await super().run(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                session=session,
            )
            current_trace.output = str(res)
            observer.result = str(res)
        return res

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        *,
        context: TContext | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        session: Session | None = None,
    ) -> RunResult:
        if run_config is None:
            run_config = RunConfig()

        if run_config.model_provider is not None:
            run_config.model_provider = _ObservedProvider(
                run_config.model_provider,
                metrics=getattr(starting_agent, "metrics", None),
                metric_collection=getattr(starting_agent, "metric_collection", None),
                deepeval_prompt=getattr(starting_agent, "deepeval_prompt", None),
            )

        input_val = input
        metrics = getattr(starting_agent, "metrics", None)
        metric_collection = getattr(starting_agent, "metric_collection", None)

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
            res = super().run_sync(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                session=session,
            )
            current_trace.output = str(res)
            observer.result = str(res)

        return res