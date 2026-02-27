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
        ) as observer:
            result = original_kickoff(self, *args, **kwargs)
            observer.result = str(result) if result else None

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
        ) as observer:
            result = original_kickoff_for_each(self, *args, **kwargs)
            observer.result = str(result) if result else None

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
        ) as observer:
            result = await original_kickoff_async(self, *args, **kwargs)
            observer.result = str(result) if result else None

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
        ) as observer:
            result = await original_kickoff_for_each_async(
                self, *args, **kwargs
            )
            observer.result = str(result) if result else None

        return result

    Crew.kickoff_for_each_async = wrapper


def wrap_crew_akickoff():
    if not hasattr(Crew, "akickoff"):
        return

    original_akickoff = Crew.akickoff

    @wraps(original_akickoff)
    async def wrapper(self, *args, **kwargs):
        metric_collection, metrics = _check_metrics_and_metric_collection(self)
        with Observer(
            span_type="crew",
            func_name="akickoff",
            metric_collection=metric_collection,
            metrics=metrics,
        ) as observer:
            result = await original_akickoff(self, *args, **kwargs)
            observer.result = str(result) if result else None

        return result

    Crew.akickoff = wrapper


def wrap_crew_akickoff_for_each():
    if not hasattr(Crew, "akickoff_for_each"):
        return

    original_akickoff_for_each = Crew.akickoff_for_each

    @wraps(original_akickoff_for_each)
    async def wrapper(self, *args, **kwargs):
        metric_collection, metrics = _check_metrics_and_metric_collection(self)
        with Observer(
            span_type="crew",
            func_name="akickoff_for_each",
            metric_collection=metric_collection,
            metrics=metrics,
        ) as observer:
            result = await original_akickoff_for_each(self, *args, **kwargs)
            observer.result = str(result) if result else None

        return result

    Crew.akickoff_for_each = wrapper


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
        ) as observer:
            result = original_execute_task(self, *args, **kwargs)
            observer.result = str(result) if result else None
        return result

    Agent.execute_task = wrapper


def wrap_agent_aexecute_task():
    if not hasattr(Agent, "aexecute_task"):
        return

    original_aexecute_task = Agent.aexecute_task

    @wraps(original_aexecute_task)
    async def wrapper(self, *args, **kwargs):
        metric_collection, metrics = _check_metrics_and_metric_collection(self)
        with Observer(
            span_type="agent",
            func_name="aexecute_task",
            metric_collection=metric_collection,
            metrics=metrics,
        ) as observer:
            result = await original_aexecute_task(self, *args, **kwargs)
            observer.result = str(result) if result else None
        return result

    Agent.aexecute_task = wrapper


def _check_metrics_and_metric_collection(obj: Any):
    metric_collection = getattr(obj, "_metric_collection", None)
    metrics = getattr(obj, "_metrics", None)
    return metric_collection, metrics
