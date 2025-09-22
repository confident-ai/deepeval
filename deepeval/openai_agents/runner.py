from __future__ import annotations

from dataclasses import replace
from typing import List, Any

from agents import (
    RunConfig,
    RunResult,
    RunResultStreaming,
    Runner as AgentsRunner,
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


class Runner(AgentsRunner):
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
        name: str | None = None,
        tags: List[str] | None = None,
        metadata: dict | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        **kwargs, # backwards compatibility
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
            update_trace_attributes(
                input=input,
                name=name,
                tags=tags,
                metadata=metadata,
                thread_id=thread_id,
                user_id=user_id,
                metric_collection=metric_collection,
                metrics=metrics,
            )
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
                **kwargs, # backwards compatibility
            )
            update_trace_attributes(output=str(res))
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

        metrics: List[BaseMetric] | None = None,
        metric_collection: str | None = None,
        name: str | None = None,
        tags: List[str] | None = None,
        metadata: dict | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        **kwargs,
    ) -> RunResult:
        if run_config is None:
            run_config = RunConfig()

        if run_config.model_provider is not None:
            run_config.model_provider = _ObservedProvider(
                run_config.model_provider,
                metrics=getattr(starting_agent, "llm_metrics", None),
                metric_collection=getattr(starting_agent, "llm_metric_collection", None),
                deepeval_prompt=getattr(starting_agent, "deepeval_prompt", None),
            )

        input_val = input

        update_trace_attributes(
            input=input_val,
            name=name,
            tags=tags,
            metadata=metadata,
            thread_id=thread_id,
            user_id=user_id,
            metric_collection=metric_collection,
            metrics=metrics,
        )

        with Observer(
            span_type="custom",
            metric_collection=metric_collection,
            metrics=metrics,
            func_name="run_sync",
            function_kwargs={"input": input_val},
        ) as observer:
            current_span = current_span_context.get()
            current_trace = current_trace_context.get()
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
                **kwargs, # backwards compatibility
            )
            update_trace_attributes(output=str(res))
            observer.result = str(res)

        return res
    
    @classmethod
    def run_streamed(
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
        name: str | None = None,
        tags: List[str] | None = None,
        metadata: dict | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        **kwargs, # backwards compatibility
    ) -> RunResultStreaming:

        if run_config is None:
            run_config = RunConfig()

        if run_config.model_provider is not None:
            run_config.model_provider = _ObservedProvider(
                run_config.model_provider,
                metrics=getattr(starting_agent, "llm_metrics", None),
                metric_collection=getattr(starting_agent, "llm_metric_collection", None),
                deepeval_prompt=getattr(starting_agent, "deepeval_prompt", None),
            )

        update_trace_attributes(
            input=input,
            name=name,
            tags=tags,
            metadata=metadata,
            thread_id=thread_id,
            user_id=user_id,
            metric_collection=metric_collection,
            metrics=metrics,
        )

        input_val = input

        # Manually manage observer lifecycle so it only closes when streaming completes
        observer = Observer(
            span_type="custom",
            metric_collection=metric_collection,
            metrics=metrics,
            func_name="run_streamed",
            function_kwargs={"input": input_val},
        )
        observer.__enter__()

        current_span = current_span_context.get()
        current_trace = current_trace_context.get()
        if current_span:
            current_span.input = input_val

        res = super().run_streamed(
            starting_agent,
            input,
            context=context,
            max_turns=max_turns,
            hooks=hooks,
            run_config=run_config,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            session=session,
            **kwargs, # backwards compatibility
        )

        # Defer setting trace output and closing observer until stream completes
        class RunResultStreamingProxy:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def __str__(self):
                return str(self._inner)

            async def stream_events(self):
                try:
                    async for event in self._inner.stream_events():
                        yield event
                    # normal completion
                    update_trace_attributes(output=str(self._inner))
                    observer.result = str(self._inner)
                    observer.__exit__(None, None, None)
                except Exception as e:
                    observer.__exit__(type(e), e, e.__traceback__)
                    raise

        return RunResultStreamingProxy(res)

def update_trace_attributes(
    input: Any | None = None,
    output: Any | None = None,
    name: str | None = None,
    tags: List[str] | None = None,
    metadata: dict | None = None,
    thread_id: str | None = None,
    user_id: str | None = None,
    metric_collection: str | None = None,
    metrics: List[BaseMetric] | None = None,
):
    current_trace = current_trace_context.get()
    if input:
        current_trace.input = input
    if output:
        current_trace.output = output
    if name:
        current_trace.name = name
    if tags:
        current_trace.tags = tags
    if metadata:
        current_trace.metadata = metadata
    if thread_id:
        current_trace.thread_id = thread_id
    if user_id:
        current_trace.user_id = user_id
    if metric_collection:
        current_trace.metric_collection = metric_collection
    if metrics:
        current_trace.metrics = metrics