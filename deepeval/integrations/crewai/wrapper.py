from functools import wraps
from typing import Any
from contextvars import copy_context

from deepeval.tracing.tracing import Observer
from .subs import try_import_core_classes
from .context_registration import CONTEXT_REG
from .identifiers import agent_exec_id

Crew, Agent, LLM = try_import_core_classes()


def _as_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        v = obj.get("raw")
        if isinstance(v, str):
            return v
    return str(obj)


def wrap_crew_kickoff():
    original_kickoff = Crew.kickoff

    @wraps(original_kickoff)
    def wrapper(self, *args, **kwargs):
        metric_collection, metrics = _check_metrics_and_metric_collection(self)
        with Observer(
            span_type="crew",
            func_name="kickoff",
            metric_collection=metric_collection,
            metrics=metrics,
        ) as _observer:
            result = original_kickoff(self, *args, **kwargs)
            _observer.result = _as_text(result)
        return result

    Crew.kickoff = wrapper


def wrap_crew_kickoff_for_each():
    original_kickoff_for_each = Crew.kickoff_for_each

    @wraps(original_kickoff_for_each)
    def wrapper(self, *args, **kwargs):
        metric_collection, metrics = _check_metrics_and_metric_collection(self)
        with Observer(
            span_type="crew",
            func_name="kickoff_for_each",
            metric_collection=metric_collection,
            metrics=metrics,
        ) as _observer:
            result = original_kickoff_for_each(self, *args, **kwargs)
            _observer.result = _as_text(result)

        return result

    Crew.kickoff_for_each = wrapper


def wrap_crew_kickoff_async():
    original_kickoff_async = Crew.kickoff_async

    @wraps(original_kickoff_async)
    async def wrapper(self, *args, **kwargs):
        metric_collection, metrics = _check_metrics_and_metric_collection(self)
        with Observer(
            span_type="crew",
            func_name="kickoff_async",
            metric_collection=metric_collection,
            metrics=metrics,
        ) as _observer:
            result = await original_kickoff_async(self, *args, **kwargs)
            _observer.result = _as_text(result)

        return result

    Crew.kickoff_async = wrapper


def wrap_crew_kickoff_for_each_async():
    original_kickoff_for_each_async = Crew.kickoff_for_each_async

    @wraps(original_kickoff_for_each_async)
    async def wrapper(self, *args, **kwargs):
        metric_collection, metrics = _check_metrics_and_metric_collection(self)
        with Observer(
            span_type="crew",
            func_name="kickoff_for_each_async",
            metric_collection=metric_collection,
            metrics=metrics,
        ) as _observer:
            result = await original_kickoff_for_each_async(
                self, *args, **kwargs
            )
            _observer.result = _as_text(result)
        return result

    Crew.kickoff_for_each_async = wrapper


def wrap_llm_call():
    original_llm_call = LLM.call

    @wraps(original_llm_call)
    def wrapper(self, *args, **kwargs):
        metric_collection, metrics = _check_metrics_and_metric_collection(self)
        with Observer(
            span_type="llm",
            func_name="call",
            observe_kwargs={"model": "temp_model"},
            metric_collection=metric_collection,
            metrics=metrics,
        ) as _observer:
            result = original_llm_call(self, *args, **kwargs)
            _observer.result = _as_text(result)

    LLM.call = wrapper


def wrap_agent_execute_task():
    original_execute_task = Agent.execute_task

    @wraps(original_execute_task)
    def wrapper(self, *args, **kwargs):
        metric_collection, metrics = _check_metrics_and_metric_collection(self)
        with Observer(
            span_type="agent",
            func_name="execute_task",
            metric_collection=metric_collection,
            metrics=metrics,
        ) as _observer:
            # capture a context that includes the Agent span binding
            # used by AgentExecutionStartedEvent and AgentExecutionCompletedEvent handlers
            CONTEXT_REG.agent.set(agent_exec_id(self), copy_context())
            result = original_execute_task(self, *args, **kwargs)
            # feed result into span via Observer
            _observer.result = _as_text(result)
        return result

    Agent.execute_task = wrapper


def _check_metrics_and_metric_collection(obj: Any):
    metric_collection = getattr(obj, "_metric_collection", None)
    metrics = getattr(obj, "_metrics", None)
    return metric_collection, metrics
