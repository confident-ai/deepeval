import inspect
from typing import Optional, List
from contextvars import ContextVar

from deepeval.prompt import Prompt
from deepeval.tracing.types import AgentSpan
from deepeval.tracing.tracing import Observer
from deepeval.metrics.base_metric import BaseMetric
from deepeval.tracing.context import current_span_context
from deepeval.integrations.pydantic_ai.utils import extract_tools_called

try:
    from pydantic_ai.agent import Agent
    from deepeval.integrations.pydantic_ai.utils import create_patched_tool, update_trace_context, patch_llm_model
    is_pydantic_ai_installed = True
except:
    is_pydantic_ai_installed = False
    
def pydantic_ai_installed():
    if not is_pydantic_ai_installed:
        raise ImportError(
            "Pydantic AI is not installed. Please install it with `pip install pydantic-ai`."
        )

_IS_RUN_SYNC = ContextVar("deepeval_is_run_sync", default=False)   

class DeepEvalPydanticAIAgent(Agent):

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
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metric_collection: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None, 
        llm_metric_collection: Optional[str] = None,
        llm_metrics: Optional[List[BaseMetric]] = None,
        llm_prompt: Optional[Prompt] = None,
        agent_metric_collection: Optional[str] = None,
        agent_metrics: Optional[List[BaseMetric]] = None,
        **kwargs
    ):
        pydantic_ai_installed()

        self.trace_name = name
        self.trace_tags = tags
        self.trace_metadata = metadata
        self.trace_thread_id = thread_id
        self.trace_user_id = user_id
        self.trace_metric_collection = metric_collection
        self.trace_metrics = metrics
        
        self.llm_metric_collection = llm_metric_collection
        self.llm_metrics = llm_metrics
        self.llm_prompt = llm_prompt
        
        self.agent_metric_collection = agent_metric_collection
        self.agent_metrics = agent_metrics
        
        super().__init__(*args, **kwargs)
        
        patch_llm_model(self._model, llm_metric_collection, llm_metrics, llm_prompt) #TODO: Add dual patch guards
        

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
    ):
        sig = inspect.signature(super().run)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        input = bound.arguments.get("user_prompt", None)
        
        with Observer(
            span_type="agent" if not _IS_RUN_SYNC.get() else "custom",
            func_name="Agent" if not _IS_RUN_SYNC.get() else "run",
            function_kwargs={"input": input},
            metrics=self.agent_metrics if not _IS_RUN_SYNC.get() else None,
            metric_collection=self.agent_metric_collection if not _IS_RUN_SYNC.get() else None,
        ) as observer:
            result = await super().run(*args, **kwargs)
            observer.result = result.output
            update_trace_context(
                trace_name=self.name if self.name else name,
                trace_tags=self.trace_tags if self.trace_tags else tags,
                trace_metadata=self.trace_metadata if self.trace_metadata else metadata,
                trace_thread_id=self.trace_thread_id if self.trace_thread_id else thread_id,
                trace_user_id=self.trace_user_id if self.trace_user_id else user_id,
                trace_metric_collection=self.trace_metric_collection if self.trace_metric_collection else metric_collection,
                trace_metrics=self.trace_metrics if self.trace_metrics else metrics,
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
    ):
        sig = inspect.signature(super().run_sync)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        input = bound.arguments.get("user_prompt", None)
        
        token = _IS_RUN_SYNC.set(True)

        with Observer(
            span_type="agent",
            func_name="Agent",
            function_kwargs={"input": input},
            metrics=self.agent_metrics,
            metric_collection=self.agent_metric_collection,
        ) as observer:
            try:
                result = super().run_sync(*args, **kwargs)
            finally:
                _IS_RUN_SYNC.reset(token)

            observer.result = result.output
            update_trace_context(
                trace_name=self.name if self.name else name,
                trace_tags=self.tags if self.tags else tags,
                trace_metadata=self.metadata if self.metadata else metadata,
                trace_thread_id=self.thread_id if self.thread_id else thread_id,
                trace_user_id=self.user_id if self.user_id else user_id,
                trace_metric_collection=self.trace_metric_collection if self.trace_metric_collection else metric_collection,
                trace_metrics=self.trace_metrics if self.trace_metrics else metrics,
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
        
    
    def tool(
        self,
        *args,
        metrics: Optional[List[BaseMetric]] = None,
        metric_collection: Optional[str] = None,
        **kwargs
    ):
        # Direct decoration: @agent.tool
        if args and callable(args[0]):
            patched_func = create_patched_tool(args[0], metrics, metric_collection)
            new_args = (patched_func,) + args[1:]
            return super(DeepEvalPydanticAIAgent, self).tool(*new_args, **kwargs)
        # Decoration with args: @agent.tool(...)
        super_tool = super(DeepEvalPydanticAIAgent, self).tool
        def decorator(func):
            patched_func = create_patched_tool(func, metrics, metric_collection)
            return super_tool(*args, **kwargs)(patched_func)
        return decorator