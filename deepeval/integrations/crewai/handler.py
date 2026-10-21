"""deepeval's CrewAI integration, built on confident-trace.

confident-trace traces crews, tasks, agents and tools once deepeval starts it
with the ``crewai`` instrumentation, and their model calls with the provider
ones. deepeval adds a span for each knowledge retrieval; its ``Crew``,
``Agent``, ``LLM`` and ``tool`` add the metrics they are given.
"""

import functools
import inspect
from typing import Any, Callable, Optional

from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.context import update_current_span
from deepeval.tracing.integrations import Integration
from deepeval.tracing.otel.setup import ensure_confident_trace

try:
    import crewai  # noqa: F401

    crewai_installed = True
except ImportError:
    crewai_installed = False

# CrewAI's model calls are traced through their provider's SDK.
INSTRUMENTATIONS = (
    "crewai",
    "openai",
    "anthropic",
    "google_genai",
    "bedrock",
    "litellm",
)
_RETRIEVALS = ("handle_knowledge_retrieval", "ahandle_knowledge_retrieval")
_FOR_EACH = ("kickoff_for_each", "kickoff_for_each_async", "akickoff_for_each")


def is_crewai_installed():
    if not crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )


def instrument_crewai(api_key: Optional[str] = None):
    is_crewai_installed()
    with capture_tracing_integration(Integration.CREW_AI):
        ensure_confident_trace(INSTRUMENTATIONS, api_key=api_key)
        _trace_knowledge_retrieval()
        _trace_kickoff_for_each()


def reset_crewai_instrumentation():
    """Kept for compatibility: deepeval keeps no CrewAI state of its own."""


def _trace_knowledge_retrieval() -> None:
    import crewai.agent.core as core

    for name in _RETRIEVALS:
        retrieve = getattr(core, name, None)
        if retrieve is not None and not getattr(
            retrieve, "_deepeval_patched", False
        ):
            setattr(core, name, _with_retrieval_span(retrieve))


def _with_retrieval_span(retrieve: Callable[..., Any]) -> Callable[..., Any]:
    """``retrieve``, recorded as a retriever span when the agent has knowledge.

    It returns the task prompt with what it retrieved appended.
    """
    import confident_trace

    if inspect.iscoroutinefunction(retrieve):

        @functools.wraps(retrieve)
        async def traced(agent, task, task_prompt, *args, **kwargs):
            if not _uses_knowledge(agent):
                return await retrieve(agent, task, task_prompt, *args, **kwargs)
            with confident_trace.span(
                name="knowledge_retrieval", type="retriever"
            ):
                prompt = await retrieve(
                    agent, task, task_prompt, *args, **kwargs
                )
                _record_retrieval(agent, task_prompt, prompt)
                return prompt

    else:

        @functools.wraps(retrieve)
        def traced(agent, task, task_prompt, *args, **kwargs):
            if not _uses_knowledge(agent):
                return retrieve(agent, task, task_prompt, *args, **kwargs)
            with confident_trace.span(
                name="knowledge_retrieval", type="retriever"
            ):
                prompt = retrieve(agent, task, task_prompt, *args, **kwargs)
                _record_retrieval(agent, task_prompt, prompt)
                return prompt

    traced._deepeval_patched = True
    return traced


def _trace_kickoff_for_each() -> None:
    """Run each ``kickoff_for_each`` under one span, carrying the crew's
    metrics: CrewAI kicks off a copy of the crew per input."""
    from crewai import Crew

    from deepeval.integrations.crewai.subs import with_staged_metrics

    for name in _FOR_EACH:
        kickoff = getattr(Crew, name, None)
        if kickoff is not None and not getattr(
            kickoff, "_deepeval_patched", False
        ):
            spanned = _with_span(kickoff, name)
            traced = with_staged_metrics(lambda spanned=spanned: spanned, None)
            traced._deepeval_patched = True
            setattr(Crew, name, traced)


def _with_span(method: Callable[..., Any], name: str) -> Callable[..., Any]:
    """``method``, run inside a span called ``name``."""
    import confident_trace

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def traced(*args, **kwargs):
            with confident_trace.span(name=name):
                return await method(*args, **kwargs)

    else:

        @functools.wraps(method)
        def traced(*args, **kwargs):
            with confident_trace.span(name=name):
                return method(*args, **kwargs)

    return traced


def _uses_knowledge(agent: Any) -> bool:
    # CrewAI's own test for whether it retrieves anything.
    crew = agent.crew if not isinstance(agent.crew, str) else None
    return bool(agent.knowledge or (crew and crew.knowledge))


def _record_retrieval(agent: Any, task_prompt: Any, prompt: Any) -> None:
    retrieved = (
        prompt[len(task_prompt) :]
        if isinstance(prompt, str)
        and isinstance(task_prompt, str)
        and prompt.startswith(task_prompt)
        else None
    )
    update_current_span(
        input=getattr(agent, "knowledge_search_query", None),
        output=retrieved or None,
        retrieval_context=[retrieved] if retrieved else None,
    )
