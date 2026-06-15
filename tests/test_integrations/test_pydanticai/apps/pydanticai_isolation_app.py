"""PydanticAI Isolation App: behavioral validation of contextvar isolation
across concurrent ``asyncio.gather`` tasks and across threads in a
``ThreadPoolExecutor``.

Mirrors ``pydantic_after_concurrent.py`` and ``pydantic_after_threads.py``,
distilled to a pytest-runnable form.

**No schema files for these tests.** ``trace_testing_manager.test_dict``
is a single global slot that gets overwritten by every ``end_trace`` call,
so when N concurrent traces complete, only the last winner is captured
(racily). These tests instead assert the **isolation invariant** in
user-space: each task / thread sees its own ``_request_ctx`` contents
both before AND after ``agent.run`` returns, and no two tasks ever
observe the same ``request_id``. That's the property the validation
scripts exist to prove; trace-shape isn't the relevant signal.

If full per-trace shape validation across concurrent runs is ever
needed, ``trace_testing_manager`` would have to grow a multi-trace
capture path (list-of-dicts keyed by trace UUID, plus a
``wait_for_test_dicts(n)`` waiter, plus a multi-schema decorator).
That's a follow-up; the isolation invariant is already covered here.
"""

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from pydantic_ai import Agent

from deepeval.integrations.pydantic_ai import DeepEvalInstrumentationSettings
from deepeval.tracing import update_current_span, update_current_trace


# Per-request ContextVar carrying request data the tool body reads back.
# In each task / worker thread we ``set`` this BEFORE calling agent.run;
# inside the tool we ``get`` it. The whole point is to confirm the value
# we get back is the one WE set — never another task's / thread's.
_request_ctx: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "_pydanticai_test_request_ctx", default={}
)


def create_isolation_agent(
    name: str = "pydanticai-isolation-test",
) -> Agent:
    """Agent with one tool that reads ``_request_ctx`` and writes it
    onto both the tool span and the (implicit) trace, so an outside
    observer can verify per-task / per-thread isolation."""
    settings = DeepEvalInstrumentationSettings(name=name)

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt=(
            "You are an assistant. When the user asks for data with a "
            "specific key, call the get_data tool with that key. Be concise."
        ),
        instrument=settings,
        name="isolation_agent",
    )

    @agent.tool_plain
    def get_data(key: str) -> str:
        """Read the per-request contextvar and stamp it onto the
        tool span + implicit trace. Returns a stable, key-derived
        string so each task's output is distinguishable."""
        req = _request_ctx.get()
        request_id = req.get("request_id")
        user_id = req.get("user_id")

        update_current_span(
            metadata={
                "request_id_from_ctx": request_id,
                "user_id_from_ctx": user_id,
                "key": key,
                "thread_name": threading.current_thread().name,
            },
        )
        update_current_trace(
            user_id=user_id,
            metadata={
                "request_id": request_id,
                "thread_name": threading.current_thread().name,
            },
        )

        return f"data-for-{key}"

    return agent


# Each request: (prompt, user_id, request_id, expected_key_in_output)
RequestSpec = Tuple[str, str, str, str]


def _build_prompt(key: str) -> str:
    return f"Use the get_data tool with key='{key}' and report the result."


def make_distinct_requests() -> List[RequestSpec]:
    """Three distinct request specs. Used by both the concurrent and
    threaded helpers to drive identical isolation checks."""
    return [
        (_build_prompt("alpha"), "user-a", "req-iso-001", "alpha"),
        (_build_prompt("beta"), "user-b", "req-iso-002", "beta"),
        (_build_prompt("gamma"), "user-c", "req-iso-003", "gamma"),
    ]


async def concurrent_isolation_run(
    agent: Agent,
    requests: List[RequestSpec],
) -> List[Dict[str, Any]]:
    """Fire N ``await agent.run(...)`` calls via ``asyncio.gather``.
    Each task sets ``_request_ctx`` to its own values before the call
    and re-reads it afterwards to verify intra-task stability."""

    async def _one(
        prompt: str, user_id: str, request_id: str, expected_key: str
    ) -> Dict[str, Any]:
        _request_ctx.set({"user_id": user_id, "request_id": request_id})
        result = await agent.run(prompt)
        post_run = _request_ctx.get()
        return {
            "user_id": user_id,
            "request_id": request_id,
            "expected_key": expected_key,
            "output": result.output,
            "post_run_user_id": post_run.get("user_id"),
            "post_run_request_id": post_run.get("request_id"),
        }

    return await asyncio.gather(*(_one(p, u, r, k) for p, u, r, k in requests))


def threaded_isolation_run(
    agent: Agent,
    requests: List[RequestSpec],
) -> List[Dict[str, Any]]:
    """Fire N ``agent.run_sync(...)`` calls from worker threads via
    ``ThreadPoolExecutor``. Each worker establishes its own
    ``_request_ctx`` (TPE does NOT inherit contextvars from the
    submitting thread by default) and re-reads it after the call."""

    def _one(
        prompt: str, user_id: str, request_id: str, expected_key: str
    ) -> Dict[str, Any]:
        _request_ctx.set({"user_id": user_id, "request_id": request_id})
        result = agent.run_sync(prompt)
        post_run = _request_ctx.get()
        return {
            "user_id": user_id,
            "request_id": request_id,
            "expected_key": expected_key,
            "output": result.output,
            "post_run_user_id": post_run.get("user_id"),
            "post_run_request_id": post_run.get("request_id"),
            "thread_name": threading.current_thread().name,
        }

    with ThreadPoolExecutor(
        max_workers=len(requests),
        thread_name_prefix="pydanticai-isolation-worker",
    ) as pool:
        futures = [pool.submit(_one, p, u, r, k) for p, u, r, k in requests]
        return [f.result() for f in futures]
