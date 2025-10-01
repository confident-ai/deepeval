from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, List
from deepeval.tracing.context import current_span_context
from deepeval.tracing.types import AgentSpan, ToolSpan
from deepeval.tracing.utils import make_json_serializable
from deepeval.tracing import observe
from deepeval.tracing.tracing import Observer
from deepeval.metrics import BaseMetric
from deepeval.prompt import Prompt
from deepeval.tracing.types import LlmSpan
from functools import wraps

try:
    from agents import function_tool as _agents_function_tool  # type: ignore
    from deepeval.openai_agents.extractors import parse_response_output
    from agents.run import AgentRunner
    from agents.run import SingleStepResult
    from agents.models.interface import Model
    from agents import Agent
except Exception:
    pass


def _wrap_with_observe(
    func: Callable[..., Any],
    metrics: Optional[str] = None,
    metric_collection: Optional[str] = None,
) -> Callable[..., Any]:
    if getattr(func, "_is_deepeval_observed", False):
        return func

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def observed(*args: Any, **kwargs: Any) -> Any:
            current_span = current_span_context.get()
            if isinstance(current_span, ToolSpan):
                current_span.metrics = metrics
                current_span.metric_collection = metric_collection
            return await func(*args, **kwargs)

    else:

        @wraps(func)
        def observed(*args: Any, **kwargs: Any) -> Any:
            current_span = current_span_context.get()
            if isinstance(current_span, ToolSpan):
                current_span.metrics = metrics
                current_span.metric_collection = metric_collection
            return func(*args, **kwargs)

    setattr(observed, "_is_deepeval_observed", True)
    try:
        observed.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    except Exception:
        pass
    return observed


def function_tool(
    func: Optional[Callable[..., Any]] = None, /, *args: Any, **kwargs: Any
) -> Any:
    metrics = kwargs.pop("metrics", None)
    metric_collection = kwargs.pop("metric_collection", None)

    if _agents_function_tool is None:
        raise RuntimeError(
            "agents.function_tool is not available. Please install agents via your package manager"
        )

    if callable(func):

        wrapped = _wrap_with_observe(
            func,
            metrics=metrics,
            metric_collection=metric_collection,
        )
        return _agents_function_tool(wrapped, *args, **kwargs)

    def decorator(real_func: Callable[..., Any]) -> Any:

        wrapped = _wrap_with_observe(
            real_func,
            metrics=metrics,
            metric_collection=metric_collection,
        )
        return _agents_function_tool(wrapped, *args, **kwargs)

    return decorator


_PATCHED_DEFAULT_RUN_SINGLE_TURN = False
_PATCHED_DEFAULT_RUN_SINGLE_TURN_STREAMED = False
_PATCHED_DEFAULT_GET_MODEL = False


class _ObservedModel(Model):
    def __init__(
        self,
        inner: Model,
        llm_metric_collection: str = None,
        llm_metrics: List[BaseMetric] = None,
        confident_prompt: Prompt = None,
    ) -> None:
        self._inner = inner
        self._llm_metric_collection = llm_metric_collection
        self._llm_metrics = llm_metrics
        self._confident_prompt = confident_prompt

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def get_response(
        self,
        *args,
        **kwargs,
    ):
        with Observer(
            span_type="llm",
            func_name="LLM",
            observe_kwargs={"model": "temp_model"},
            metrics=self._llm_metrics,
            metric_collection=self._llm_metric_collection,
        ):
            result = await self._inner.get_response(
                *args,
                **kwargs,
            )
            llm_span: LlmSpan = current_span_context.get()
            llm_span.prompt = self._confident_prompt

        return result

    def stream_response(
        self,
        *args,
        **kwargs,
    ):

        async def _gen():
            observer = Observer(
                span_type="llm",
                func_name="LLM",
                observe_kwargs={"model": "temp_model"},
                metrics=self._llm_metrics,
                metric_collection=self._llm_metric_collection,
            )
            observer.__enter__()

            llm_span: LlmSpan = current_span_context.get()
            llm_span.prompt = self._confident_prompt

            try:
                async for event in self._inner.stream_response(
                    *args,
                    **kwargs,
                ):
                    yield event
            except Exception as e:
                observer.__exit__(type(e), e, e.__traceback__)
                raise
            finally:
                observer.__exit__(None, None, None)

        return _gen()


def patch_default_agent_run_single_turn():
    global _PATCHED_DEFAULT_RUN_SINGLE_TURN
    if _PATCHED_DEFAULT_RUN_SINGLE_TURN:
        return

    original_run_single_turn = AgentRunner._run_single_turn

    @classmethod
    async def patched_run_single_turn(cls, *args, **kwargs):
        res: SingleStepResult = await original_run_single_turn.__func__(
            cls, *args, **kwargs
        )
        try:
            if isinstance(res, SingleStepResult):
                agent_span = current_span_context.get()
                if isinstance(agent_span, AgentSpan):
                    _set_agent_metrics(kwargs.get("agent", None), agent_span)
                    if agent_span.input is None:
                        _pre_step_items_raw_list = [
                            item.raw_item for item in res.pre_step_items
                        ]
                        agent_span.input = (
                            make_json_serializable(_pre_step_items_raw_list)
                            if _pre_step_items_raw_list
                            else make_json_serializable(res.original_input)
                        )
                    agent_span.output = parse_response_output(
                        res.model_response.output
                    )
        except Exception:
            pass
        return res

    AgentRunner._run_single_turn = patched_run_single_turn
    _PATCHED_DEFAULT_RUN_SINGLE_TURN = True  # type: ignore


def patch_default_agent_run_single_turn_streamed():
    global _PATCHED_DEFAULT_RUN_SINGLE_TURN_STREAMED
    if _PATCHED_DEFAULT_RUN_SINGLE_TURN_STREAMED:
        return

    original_run_single_turn_streamed = AgentRunner._run_single_turn_streamed

    @classmethod
    async def patched_run_single_turn_streamed(cls, *args, **kwargs):

        res: SingleStepResult = (
            await original_run_single_turn_streamed.__func__(
                cls, *args, **kwargs
            )
        )
        try:
            if isinstance(res, SingleStepResult):
                agent_span = current_span_context.get()
                if isinstance(agent_span, AgentSpan):
                    _set_agent_metrics(
                        kwargs.get("agent", None), agent_span
                    )  # TODO: getting no agent
                    if agent_span.input is None:
                        _pre_step_items_raw_list = [
                            item.raw_item for item in res.pre_step_items
                        ]
                        agent_span.input = (
                            make_json_serializable(_pre_step_items_raw_list)
                            if _pre_step_items_raw_list
                            else make_json_serializable(res.original_input)
                        )
                    agent_span.output = parse_response_output(
                        res.model_response.output
                    )
        except Exception:
            pass
        return res

    AgentRunner._run_single_turn_streamed = patched_run_single_turn_streamed
    _PATCHED_DEFAULT_RUN_SINGLE_TURN_STREAMED = True  # type: ignore


def patch_default_agent_runner_get_model():
    global _PATCHED_DEFAULT_GET_MODEL
    if _PATCHED_DEFAULT_GET_MODEL:
        return

    original_get_model_cm = AgentRunner._get_model
    try:
        original_get_model = original_get_model_cm.__func__
    except AttributeError:
        original_get_model = (
            original_get_model_cm  # fallback (non-classmethod edge case)
        )

    def patched_get_model(cls, *args, **kwargs) -> Model:
        model = original_get_model(cls, *args, **kwargs)

        agent = (
            kwargs.get("agent")
            if "agent" in kwargs
            else (args[0] if args else None)
        )
        if agent is None:
            return model

        if isinstance(model, _ObservedModel):
            return model

        llm_metrics = getattr(agent, "llm_metrics", None)
        llm_metric_collection = getattr(agent, "llm_metric_collection", None)
        confident_prompt = getattr(agent, "confident_prompt", None)
        return _ObservedModel(
            inner=model,
            llm_metric_collection=llm_metric_collection,
            llm_metrics=llm_metrics,
            confident_prompt=confident_prompt,
        )

    # Preserve basic metadata and mark as patched
    patched_get_model.__name__ = original_get_model.__name__
    patched_get_model.__doc__ = original_get_model.__doc__

    AgentRunner._get_model = classmethod(patched_get_model)
    _PATCHED_DEFAULT_GET_MODEL = True


def _set_agent_metrics(agent: Agent, agent_span: AgentSpan) -> None:
    try:
        if agent is None or agent_span is None:
            return
        agent_metrics = getattr(agent, "agent_metrics", None)
        agent_metric_collection = getattr(
            agent, "agent_metric_collection", None
        )

        if agent_metrics is not None:
            agent_span.metrics = agent_metrics
        if agent_metric_collection is not None:
            agent_span.metric_collection = agent_metric_collection
    except Exception:
        # Be conservative: never break the run on metrics propagation
        pass
