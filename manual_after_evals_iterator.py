"""manual_after_evals_iterator.py — manual-instrumentation analog of
``pydantic_after_evals_iterator.py``.

Same shape (agent span → child LLM span, trace-level metadata, evals_iterator
+ ``next_agent_span(metrics=[...])``) but using deepeval's NATIVE
``@observe`` decorators instead of OTel-based pydantic-ai instrumentation.

The point: isolate whether the duplicate-test-cases / dropped-children
behavior we observed in the pydantic-ai run is OTel-specific, or whether
it's a fundamental issue in the evaluator framework.

Why we suspect OTel: ``ConfidentSpanExporter.export`` ends in a cleanup
loop that calls ``end_trace`` for **every** uuid in
``trace_manager.active_traces`` — not just the trace owning the span being
exported — and then ``clear_traces()``. That's safe when there's one
in-flight trace at a time, but with three concurrent ``agent.run`` tasks
the first task's cleanup will:

  - end_trace OTHER tasks' partially-built traces (pushing them into
    ``traces_to_evaluate`` empty or with only one child),
  - wipe ``active_traces``/``active_spans`` so subsequent OTel span ends
    in those tasks ``start_new_trace`` a SECOND, fresh trace under the
    same uuid,
  - that second trace also gets ``end_trace``'d and queued, producing
    a duplicate evaluation entry per affected golden.

If THIS file (with no OTel in the loop) produces a clean 3 test cases for
3 goldens — each with both agent + llm spans, each with a single set of
metric scores — then the bug is firmly in the OTel exporter's cleanup
loop and not in ``evals_iterator`` / ``_a_execute_agentic_test_case``.

If THIS file ALSO shows duplicates / dropped children, then the bug
lives somewhere shared (e.g. in the trace-test-case → main-test-case
double-add or in ``_a_evaluate_traces`` itself) and we need to widen
the fix.

Requirements:
  - ``CONFIDENT_API_KEY`` in env (or ``deepeval login``)
  - ``OPENAI_API_KEY`` in env (the *metric* still calls OpenAI to
    judge AnswerRelevancy; the agent's "LLM" call below is a
    deterministic hard-coded responder so the run is fast + isolates
    the variable to plumbing).
"""

import asyncio
import uuid
from pathlib import Path

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.tracing import (
    observe,
    update_current_span,
    update_current_trace,
)
from deepeval.tracing.context import next_agent_span


RUN_ID = f"{Path(__file__).stem}-{uuid.uuid4().hex[:8]}"


# Hard-coded responses keep this deterministic and free of provider
# variance — the goal is to test plumbing, not LLM quality. The metric
# still calls OpenAI when scoring AnswerRelevancy.
_FAKE_RESPONSES = {
    "What's 7 * 8?": "7 * 8 is 56.",
    "What's the capital of France?": "The capital of France is Paris.",
    "Name two primary colors.": "Red and blue.",
}


@observe(type="llm", model="fake-gpt")
async def fake_llm_call(prompt: str) -> str:
    """Stand-in for pydantic-ai's ``chat <model>`` LLM span.

    Decorated with ``@observe(type="llm", model=...)`` so it materializes
    as an LLM span parented under the agent span — mirroring the agent →
    llm hierarchy pydantic-ai produces natively. ``model`` is read from
    ``observe_kwargs`` at span creation time; passing it via
    ``update_current_span(...)`` raises ``TypeError`` because that helper
    is the GENERIC mutator (not LLM-typed).
    """
    # Tiny sleep just to give the trace some realistic span duration —
    # not strictly necessary for correctness.
    await asyncio.sleep(0.05)

    response = _FAKE_RESPONSES.get(prompt, "I don't know.")

    # Mirror what the OTel exporter writes onto the LLM span from
    # gen_ai attributes, so the trace shape on the dashboard matches
    # the pydantic-ai version visually.
    update_current_span(
        input=[
            {
                "role": "system",
                "content": "Be concise. Reply with one short sentence.",
            },
            {"role": "user", "content": prompt},
        ],
        output=response,
    )
    return response


@observe(type="agent", metrics=[AnswerRelevancyMetric(threshold=0.4)])
async def run_agent_observed(prompt: str) -> str:
    """Agent driver — equivalent of ``agent.run`` in the pydantic-ai
    version. Sets the same trace-level fields that
    ``DeepEvalInstrumentationSettings`` configures over there
    (``name``, ``tags``, ``metadata``) plus trace input/output, then
    delegates to ``fake_llm_call`` as a child span.
    """
    update_current_trace(
        name="manual-evals-iterator",
        tags=["manual", "evals_iterator"],
        metadata={"run_id": RUN_ID, "script": Path(__file__).stem},
        input=[{"role": "user", "content": prompt}],
    )

    response = await fake_llm_call(prompt)

    update_current_trace(output=response)
    update_current_span(
        input=[{"role": "user", "content": prompt}],
        output=response,
        # model="fake-gpt",
    )
    return response


async def run_agent(prompt: str) -> str:
    """Mirror of ``run_agent`` in ``pydantic_after_evals_iterator.py``.

    Uses ``next_agent_span(metrics=[...])`` to stage a per-call
    AnswerRelevancyMetric on the next agent-typed span. With native
    ``@observe`` the agent span IS a real ``AgentSpan`` (not an OTel
    placeholder that gets serialized + re-hydrated by the exporter), so
    the metric attaches directly and the eval pipeline runs it as a
    span-level metric.
    """
    return await run_agent_observed(prompt)
    with next_agent_span(metrics=[AnswerRelevancyMetric(threshold=0.2)]):
        return await run_agent_observed(prompt)


dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's 7 * 8?"),
        # Golden(input="What's the capital of France?"),
        # Golden(input="Name two primary colors."),
    ]
)
metric = AnswerRelevancyMetric(threshold=0.8)


for golden in dataset.evals_iterator(
    async_config=AsyncConfig(run_async=True),
    metrics=[metric],
):
    task = asyncio.create_task(run_agent(golden.input))
    dataset.evaluate(task)
