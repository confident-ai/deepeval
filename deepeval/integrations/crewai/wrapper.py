from crewai.llm import LLM
from crewai.crew import Crew
from crewai.agent import Agent
from functools import wraps
from deepeval.tracing.tracing import Observer
from typing import Any


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
        ):
            result = original_kickoff(self, *args, **kwargs)

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
        ):
            result = original_kickoff_for_each(self, *args, **kwargs)

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
        ):
            result = await original_kickoff_async(self, *args, **kwargs)

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
        ):
            result = await original_kickoff_for_each_async(
                self, *args, **kwargs
            )

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
        ):
            result = original_llm_call(self, *args, **kwargs)
        return result

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
        ):
            result = original_execute_task(self, *args, **kwargs)
        return result

    Agent.execute_task = wrapper


def _check_metrics_and_metric_collection(obj: Any):
    metric_collection = getattr(obj, "_metric_collection", None)
    metrics = getattr(obj, "_metrics", None)
    return metric_collection, metrics
