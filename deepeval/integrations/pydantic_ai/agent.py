import inspect
from typing import Optional, List, Generic, TypeVar, AsyncIterator
from typing import Any
from contextvars import ContextVar
from contextlib import asynccontextmanager

from deepeval.prompt import Prompt
from deepeval.tracing.types import AgentSpan
from deepeval.tracing.tracing import Observer
from deepeval.metrics.base_metric import BaseMetric
from deepeval.tracing.context import current_span_context
from deepeval.integrations.pydantic_ai.utils import extract_tools_called

try:
    from pydantic_ai.agent import Agent
    from pydantic_ai.tools import AgentDepsT
    from pydantic_ai.output import OutputDataT
    from deepeval.integrations.pydantic_ai.utils import (
        create_patched_tool,
        update_trace_context,
        patch_llm_model,
    )
    from pydantic_ai.output import OutputSpec
    from pydantic_ai.result import AgentRunResult, StreamedRunResult
    is_pydantic_ai_installed = True
except:
    is_pydantic_ai_installed = False


def pydantic_ai_installed():
    if not is_pydantic_ai_installed:
        raise ImportError(
            "Pydantic AI is not installed. Please install it with `pip install pydantic-ai`."
        )


_IS_RUN_SYNC = ContextVar("deepeval_is_run_sync", default=False)


AgentDepsT = TypeVar('AgentDepsT', default=None, contravariant=True)
OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
NoneType = type(None)

class DeepEvalPydanticAIAgent(Generic[AgentDepsT, OutputDataT]):
    
    agent: Agent
    
    trace_name: Optional[str] = None
    trace_tags: Optional[List[str]] = None
    trace_metadata: Optional[dict] = None
    trace_thread_id: Optional[str] = None
    trace_user_id: Optional[str] = None
    trace_metric_collection: Optional[str] = None
    trace_metrics: Optional[List[BaseMetric]] = None

    llm_prompt: Optional[Prompt] = None
    llm_metrics: Optional[List[BaseMetric]] = None
    llm_metric_collection: Optional[str] = None

    agent_metrics: Optional[List[BaseMetric]] = None
    agent_metric_collection: Optional[str] = None

    def __init__(
        self,
        *args,
        output_type: OutputSpec[OutputDataT] = str,
        deps_type: type[AgentDepsT] = NoneType,
        trace_name: Optional[str] = None,
        trace_tags: Optional[List[str]] = None,
        trace_metadata: Optional[dict] = None,
        trace_thread_id: Optional[str] = None,
        trace_user_id: Optional[str] = None,
        trace_metric_collection: Optional[str] = None,
        trace_metrics: Optional[List[BaseMetric]] = None,
        llm_metric_collection: Optional[str] = None,
        llm_metrics: Optional[List[BaseMetric]] = None,
        llm_prompt: Optional[Prompt] = None,
        agent_metric_collection: Optional[str] = None,
        agent_metrics: Optional[List[BaseMetric]] = None,
        **kwargs
    ):
        pydantic_ai_installed()

        self.trace_name = trace_name
        self.trace_tags = trace_tags
        self.trace_metadata = trace_metadata
        self.trace_thread_id = trace_thread_id
        self.trace_user_id = trace_user_id
        self.trace_metric_collection = trace_metric_collection
        self.trace_metrics = trace_metrics

        self.llm_metric_collection = llm_metric_collection
        self.llm_metrics = llm_metrics
        self.llm_prompt = llm_prompt

        self.agent_metric_collection = agent_metric_collection
        self.agent_metrics = agent_metrics

        self.agent = Agent(
            *args, 
            output_type=output_type, 
            deps_type=deps_type, 
            **kwargs
        )

        patch_llm_model(
            self.agent.model, llm_metric_collection, llm_metrics, llm_prompt
        )  # TODO: Add dual patch guards

    async def run(
        self,
        *args,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
        **kwargs
    ) -> AgentRunResult[OutputDataT]:
        sig = inspect.signature(self.agent.run)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        input = bound.arguments.get("user_prompt", None)

        agent_name = self.agent.name if self.agent.name is not None else "Agent"

        with Observer(
            span_type="agent" if not _IS_RUN_SYNC.get() else "custom",
            func_name=agent_name if not _IS_RUN_SYNC.get() else "run",
            function_kwargs={"input": input},
            metrics=self.agent_metrics if not _IS_RUN_SYNC.get() else None,
            metric_collection=(
                self.agent_metric_collection if not _IS_RUN_SYNC.get() else None
            ),
        ) as observer:
            result = await self.agent.run(
                *args, 
                **kwargs
            )

            observer.result = result.output
            update_trace_context(
                trace_name=name if name is not None else self.trace_name,
                trace_tags=tags if tags is not None else self.trace_tags,
                trace_metadata=(
                    metadata if metadata is not None else self.trace_metadata
                ),
                trace_thread_id=(
                    thread_id if thread_id is not None else self.trace_thread_id
                ),
                trace_user_id=(
                    user_id if user_id is not None else self.trace_user_id
                ),
                trace_metric_collection=(
                    metric_collection
                    if metric_collection is not None
                    else self.trace_metric_collection
                ),
                trace_metrics=(
                    metrics if metrics is not None else self.trace_metrics
                ),
                trace_input=input,
                trace_output=result.output,
            )

            agent_span: AgentSpan = current_span_context.get()
            try:
                agent_span.tools_called = extract_tools_called(result)
            except:
                pass
            # TODO: available tools
            # TODO: agent handoffs

        return result

    def run_sync(
        self,
        *args,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metric_collection: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        **kwargs
    ) -> AgentRunResult[OutputDataT]:
        sig = inspect.signature(self.agent.run_sync)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        input = bound.arguments.get("user_prompt", None)

        token = _IS_RUN_SYNC.set(True)

        agent_name = self.agent.name if self.agent.name is not None else "Agent"

        with Observer(
            span_type="agent",
            func_name=agent_name,
            function_kwargs={"input": input},
            metrics=self.agent_metrics,
            metric_collection=self.agent_metric_collection,
        ) as observer:
            try:
                result = self.agent.run_sync(*args, **kwargs)
            finally:
                _IS_RUN_SYNC.reset(token)

            observer.result = result.output
            update_trace_context(
                trace_name=name if name is not None else self.trace_name,
                trace_tags=tags if tags is not None else self.trace_tags,
                trace_metadata=(
                    metadata if metadata is not None else self.trace_metadata
                ),
                trace_thread_id=(
                    thread_id if thread_id is not None else self.trace_thread_id
                ),
                trace_user_id=(
                    user_id if user_id is not None else self.trace_user_id
                ),
                trace_metric_collection=(
                    metric_collection
                    if metric_collection is not None
                    else self.trace_metric_collection
                ),
                trace_metrics=(
                    metrics if metrics is not None else self.trace_metrics
                ),
                trace_input=input,
                trace_output=result.output,
            )

            agent_span: AgentSpan = current_span_context.get()
            try:
                agent_span.tools_called = extract_tools_called(result)
            except:
                pass

            # TODO: available tools
            # TODO: agent handoffs

        return result

    @asynccontextmanager
    async def run_stream(
        self,
        *args,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metric_collection: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        **kwargs
    ) -> AsyncIterator[StreamedRunResult[AgentDepsT, OutputDataT]]:
        sig = inspect.signature(self.agent.run_stream)
        super_params = sig.parameters
        super_kwargs = {k: v for k, v in kwargs.items() if k in super_params}
        bound = sig.bind_partial(*args, **super_kwargs)
        bound.apply_defaults()
        input = bound.arguments.get("user_prompt", None)

        agent_name = self.agent.name if self.agent.name is not None else "Agent"

        with Observer(
            span_type="agent",
            func_name=agent_name,
            function_kwargs={"input": input},
            metrics=self.agent_metrics,
            metric_collection=self.agent_metric_collection,
        ) as observer:
            final_result = None
            async with self.agent.run_stream(*args, **kwargs) as result:
                try:
                    yield result
                finally:
                    try:
                        final_result = await result.get_output()
                        observer.result = final_result
                    except Exception:
                        pass

                    update_trace_context(
                        trace_name=(
                            name if name is not None else self.trace_name
                        ),
                        trace_tags=(
                            tags if tags is not None else self.trace_tags
                        ),
                        trace_metadata=(
                            metadata
                            if metadata is not None
                            else self.trace_metadata
                        ),
                        trace_thread_id=(
                            thread_id
                            if thread_id is not None
                            else self.trace_thread_id
                        ),
                        trace_user_id=(
                            user_id
                            if user_id is not None
                            else self.trace_user_id
                        ),
                        trace_metric_collection=(
                            metric_collection
                            if metric_collection is not None
                            else self.trace_metric_collection
                        ),
                        trace_metrics=(
                            metrics
                            if metrics is not None
                            else self.trace_metrics
                        ),
                        trace_input=input,
                        trace_output=(
                            final_result if final_result is not None else None
                        ),
                    )
                    agent_span: AgentSpan = current_span_context.get()
                    try:
                        if final_result is not None:
                            agent_span.tools_called = extract_tools_called(
                                final_result
                            )
                    except:
                        pass

    def tool(
        self,
        *args,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
        **kwargs
    ) -> Any:
        # Direct decoration: @agent.tool
        if args and callable(args[0]):
            patched_func = create_patched_tool(
                args[0], metrics, metric_collection
            )
            new_args = (patched_func,) + args[1:]
            return self.agent.tool(
                *new_args, **kwargs
            )

        def decorator(func):
            patched_func = create_patched_tool(func, metrics, metric_collection)
            return self.agent.tool(*args, **kwargs)(patched_func)

        return decorator
