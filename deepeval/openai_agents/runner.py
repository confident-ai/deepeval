from __future__ import annotations

from dataclasses import replace
from typing import List, Any, Union, Optional

try:
    from agents import (
        RunConfig,
        RunResult,
        RunResultStreaming,
        Runner as AgentsRunner,
    )
    from agents.agent import Agent
    from agents.models.interface import ModelProvider
    from agents.items import TResponseInputItem
    from agents.lifecycle import RunHooks
    from agents.memory import Session
    from agents.run import DEFAULT_MAX_TURNS
    from agents.run import AgentRunner
    from agents.run_context import TContext
    from agents.models.interface import Model

    agents_available = True
except:
    agents_available = False


def is_agents_available():
    if not agents_available:
        raise ImportError(
            "agents is required for this integration. Install it via your package manager"
        )


from deepeval.tracing.tracing import Observer
from deepeval.tracing.context import current_span_context, current_trace_context

# Import observed provider/model helpers from our agent module
from deepeval.metrics import BaseMetric
from deepeval.openai_agents.agent import _ObservedModel

_PATCHED_DEFAULT_GET_MODEL = False


def _patch_default_agent_runner_get_model():
    global _PATCHED_DEFAULT_GET_MODEL
    if _PATCHED_DEFAULT_GET_MODEL:
        return

    original_get_model = AgentRunner._get_model

    @classmethod
    def patched_get_model(
        cls, agent: Agent[Any], run_config: RunConfig
    ) -> Model:
        model = original_get_model(agent, run_config)

        # Extract attributes from agent if it's a DeepEvalAgent
        llm_metrics = getattr(agent, "llm_metrics", None)
        llm_metric_collection = getattr(agent, "llm_metric_collection", None)
        confident_prompt = getattr(agent, "confident_prompt", None)
        model = _ObservedModel(
            inner=model,
            llm_metric_collection=llm_metric_collection,
            llm_metrics=llm_metrics,
            confident_prompt=confident_prompt,
        )

        return model

    # Replace the method
    AgentRunner._get_model = patched_get_model
    _PATCHED_DEFAULT_GET_MODEL = True


if agents_available:
    _patch_default_agent_runner_get_model()


class Runner(AgentsRunner):

    @classmethod
    async def run(
        cls,
        starting_agent: Agent[TContext],
        input: Union[str, list[TResponseInputItem]],
        *,
        context: Optional[TContext] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: Optional[RunHooks[TContext]] = None,
        run_config: Optional[RunConfig] = None,
        previous_response_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,  # backwards compatibility
    ) -> RunResult:
        is_agents_available()
        # _patch_default_agent_runner_get_model()

        with Observer(
            span_type="custom",
            metric_collection=metric_collection,
            metrics=metrics,
            func_name="run",
            function_kwargs={"input": input},  # also set below
        ) as observer:
            update_trace_attributes(
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
            if not current_trace.input:
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
                **kwargs,  # backwards compatibility
            )
            current_trace_thread_id = current_trace_context.get().thread_id
            _output = None
            if current_trace_thread_id:
                _output = res.final_output
            else:
                _output = str(res)
            observer.result = _output
            update_trace_attributes(output=_output)
        return res

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[TContext],
        input: Union[str, list[TResponseInputItem]],
        *,
        context: Optional[TContext] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: Optional[RunHooks[TContext]] = None,
        run_config: Optional[RunConfig] = None,
        previous_response_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> RunResult:
        is_agents_available()

        with Observer(
            span_type="custom",
            metric_collection=metric_collection,
            metrics=metrics,
            func_name="run_sync",
            function_kwargs={"input": input},  # also set below
        ) as observer:
            update_trace_attributes(
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
            if not current_trace.input:
                current_trace.input = input
            if current_span:
                current_span.input = input
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
                **kwargs,  # backwards compatibility
            )
            current_trace_thread_id = current_trace_context.get().thread_id
            _output = None
            if current_trace_thread_id:
                _output = res.final_output
            else:
                _output = str(res)
            update_trace_attributes(output=_output)
            observer.result = _output

        return res

    @classmethod
    def run_streamed(
        cls,
        starting_agent: Agent[TContext],
        input: Union[str, list[TResponseInputItem]],
        *,
        context: Optional[TContext] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        hooks: Optional[RunHooks[TContext]] = None,
        run_config: Optional[RunConfig] = None,
        previous_response_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,  # backwards compatibility
    ) -> RunResultStreaming:
        is_agents_available()
        # Manually enter observer; we'll exit when streaming finishes
        observer = Observer(
            span_type="custom",
            metric_collection=metric_collection,
            metrics=metrics,
            func_name="run_streamed",
            function_kwargs={"input": input},
        )
        observer.__enter__()

        update_trace_attributes(
            name=name,
            tags=tags,
            metadata=metadata,
            thread_id=thread_id,
            user_id=user_id,
            metric_collection=metric_collection,
            metrics=metrics,
        )
        current_trace = current_trace_context.get()
        if not current_trace.input:
            current_trace.input = input

        current_span = current_span_context.get()
        if current_span:
            current_span.input = input

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
            **kwargs,  # backwards compatibility
        )

        # Runtime-patch stream_events so the observer closes only after streaming completes
        orig_stream_events = res.stream_events

        async def _patched_stream_events(self: RunResultStreaming):
            try:
                async for event in orig_stream_events():
                    yield event
                observer.result = self.final_output
                update_trace_attributes(output=self.final_output)
            except Exception as e:
                observer.__exit__(type(e), e, e.__traceback__)
                raise
            finally:
                observer.__exit__(None, None, None)

        from types import MethodType as _MethodType

        res.stream_events = _MethodType(_patched_stream_events, res)

        return res


def update_trace_attributes(
    input: Any = None,
    output: Any = None,
    name: str = None,
    tags: List[str] = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
    metric_collection: str = None,
    metrics: List[BaseMetric] = None,
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
