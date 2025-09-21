from typing import AsyncIterator, Generic, Optional, List, Any
from contextvars import ContextVar
from contextlib import asynccontextmanager
from collections.abc import Sequence

from deepeval.prompt import Prompt
from deepeval.tracing.types import AgentSpan
from deepeval.tracing.tracing import Observer
from deepeval.metrics.base_metric import BaseMetric
from deepeval.tracing.context import current_span_context
from deepeval.integrations.pydantic_ai.utils import extract_tools_called

try:
    from pydantic_ai.agent import (
        Agent,
        EndStrategy,
        HistoryProcessor,
        EventStreamHandler,
        InstrumentationSettings,
        AgentRunResult,
    )
    from pydantic_ai.agent.abstract import RunOutputDataT
    from pydantic_ai import messages as _messages
    from pydantic_ai import usage as _usage
    from pydantic_ai.tools import (
        AgentDepsT,
        Tool,
        ToolFuncEither,
        ToolsPrepareFunc,
        DeferredToolResults,
        ToolFuncContext,
        ToolParams,
        ToolPrepareFunc,
        DocstringFormat,
        GenerateToolJsonSchema,
    )
    from pydantic_ai.toolsets import AbstractToolset
    from pydantic_ai.toolsets._dynamic import ToolsetFunc
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.builtin_tools import AbstractBuiltinTool
    from pydantic_ai import models, _system_prompt
    from pydantic_ai.output import OutputDataT, OutputSpec
    from pydantic.json_schema import GenerateJsonSchema
    from pydantic_ai.result import StreamedRunResult
    
    from deepeval.integrations.pydantic_ai.utils import create_patched_tool, update_trace_context, patch_llm_model

    is_pydantic_ai_installed = True
except:
    is_pydantic_ai_installed = False

NoneType = type(None)

def pydantic_ai_installed():
    if not is_pydantic_ai_installed:
        raise ImportError(
            "Pydantic AI is not installed. Please install it with `pip install pydantic-ai`."
        )


_IS_RUN_SYNC = ContextVar("deepeval_is_run_sync", default=False)

try:
    from typing import TypeVar
    AgentDepsT = TypeVar('AgentDepsT', default=None, covariant=True)
    OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
except TypeError:
    from typing_extensions import TypeVar
    AgentDepsT = TypeVar('AgentDepsT', default=None, covariant=True)
    OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)

class DeepEvalPydanticAIAgent(
    Agent,
    Generic[AgentDepsT, OutputDataT],  # make subclass generic
):

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
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: str
        | _system_prompt.SystemPromptFunc[AgentDepsT]
        | Sequence[str | _system_prompt.SystemPromptFunc[AgentDepsT]]
        | None = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AbstractBuiltinTool] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        
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

        **_deprecated_kwargs: Any,
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
        
        super().__init__(
            name=name,
            model=model,
            output_type=output_type,
            instructions=instructions,
            system_prompt=system_prompt,
            deps_type=deps_type,
            model_settings=model_settings,
            retries=retries,
            output_retries=output_retries,
            tools=tools,
            builtin_tools=builtin_tools,
            prepare_tools=prepare_tools,
            prepare_output_tools=prepare_output_tools,
            toolsets=toolsets,
            defer_model_check=defer_model_check,
            end_strategy=end_strategy,
            instrument=instrument,
            history_processors=history_processors,
            event_stream_handler=event_stream_handler,
            **_deprecated_kwargs,
        )
        
        patch_llm_model(self._model, llm_metric_collection, llm_metrics, llm_prompt) #TODO: Add dual patch guards
        

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
    ) -> AgentRunResult[OutputDataT]:
        input = user_prompt

        agent_name = super().name if super().name is not None else "Agent"

        with Observer(
            span_type="agent" if not _IS_RUN_SYNC.get() else "custom",
            func_name=agent_name if not _IS_RUN_SYNC.get() else "run",
            function_kwargs={"input": input},
            metrics=self.agent_metrics if not _IS_RUN_SYNC.get() else None,
            metric_collection=(
                self.agent_metric_collection if not _IS_RUN_SYNC.get() else None
            ),
        ) as observer:
            result = await super().run(
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
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
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metric_collection: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
    ) -> AgentRunResult[OutputDataT]:
        input = user_prompt
        
        token = _IS_RUN_SYNC.set(True)

        agent_name = super().name if super().name is not None else "Agent"

        with Observer(
            span_type="agent",
            func_name=agent_name,
            function_kwargs={"input": input},
            metrics=self.agent_metrics,
            metric_collection=self.agent_metric_collection,
        ) as observer:
            try:
                result = super().run_sync(
                    user_prompt=user_prompt,
                    output_type=output_type,
                    message_history=message_history,
                    deferred_tool_results=deferred_tool_results,
                    model=model,
                    deps=deps,
                    model_settings=model_settings,
                    usage_limits=usage_limits,
                    usage=usage,
                    infer_name=infer_name,
                    toolsets=toolsets,
                    event_stream_handler=event_stream_handler,
                )
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
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metric_collection: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
    ) -> AsyncIterator[StreamedRunResult[AgentDepsT, OutputDataT]]:
        input = user_prompt

        agent_name = super().name if super().name is not None else "Agent"

        with Observer(
            span_type="agent",
            func_name=agent_name,
            function_kwargs={"input": input},
            metrics=self.agent_metrics,
            metric_collection=self.agent_metric_collection,
        ) as observer:
            final_result = None
            async with super().run_stream(
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
            ) as result:
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
        func: ToolFuncContext[AgentDepsT, ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
       
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
    ) -> Any:
        # Direct decoration: @agent.tool
        if func is not None and callable(func):
            patched_func = create_patched_tool(func, metrics, metric_collection)
            return super(DeepEvalPydanticAIAgent, self).tool(
                patched_func,
                name=name,
                retries=retries,
                prepare=prepare,
                docstring_format=docstring_format,
                require_parameter_descriptions=require_parameter_descriptions,
                schema_generator=schema_generator,
                strict=strict,
                sequential=sequential,
                requires_approval=requires_approval,
                metadata=metadata,
            )
        # Decoration with args: @agent.tool(...)
        super_tool = super(DeepEvalPydanticAIAgent, self).tool

        def decorator(func_: ToolFuncContext[AgentDepsT, ToolParams]):
            patched_func = create_patched_tool(func_, metrics, metric_collection)
            return super_tool(
                name=name,
                retries=retries,
                prepare=prepare,
                docstring_format=docstring_format,
                require_parameter_descriptions=require_parameter_descriptions,
                schema_generator=schema_generator,
                strict=strict,
                sequential=sequential,
                requires_approval=requires_approval,
                metadata=metadata,
            )(patched_func)
        return decorator