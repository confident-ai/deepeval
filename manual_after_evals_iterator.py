"""Multi-agent debug script for the OTel-vs-native evals_iterator comparison.

Orchestrator → planner / retriever / calculator / knowledge / synthesizer
sub-agents, all wired with native @observe. Deterministic fake LLM + tools
so only TaskCompletionMetric calls real OpenAI.
"""

import asyncio
import uuid
from pathlib import Path

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import (
    observe,
    update_current_span,
    update_current_trace,
)
from deepeval.tracing.context import next_agent_span


RUN_ID = f"{Path(__file__).stem}-{uuid.uuid4().hex[:8]}"


_KNOWLEDGE_BASE = {
    "france": "France's capital is Paris, established under Hugh Capet in 987 CE.",
    "japan": "Japan's capital is Tokyo, renamed from Edo in 1868.",
    "primary colors": "The traditional artistic primary colors are red, yellow, and blue.",
    "rgb": "The additive primaries used in screens are red, green, and blue.",
}

_PLANS = {
    "What's 7 * 8?": ["calculator:7*8"],
    "What's the capital of France?": ["knowledge:france"],
    "Name two primary colors.": ["knowledge:primary colors"],
    "What's (12+3)*2, and the year France's capital became official?": [
        "calculator:(12+3)*2",
        "knowledge:france",
    ],
}

_CALC_RESULTS = {"7*8": "56", "(12+3)*2": "30"}

_REWRITES = {
    "What's 7 * 8?": "seven times eight",
    "What's the capital of France?": "capital of france",
    "Name two primary colors.": "primary colors list",
    "What's (12+3)*2, and the year France's capital became official?": (
        "(12+3)*2 result and year capital of france established"
    ),
}

_SYNTHESIS = {
    "What's 7 * 8?": "7 multiplied by 8 is 56.",
    "What's the capital of France?": "The capital of France is Paris (since 987 CE).",
    "Name two primary colors.": "Two primary colors are red and blue.",
    "What's (12+3)*2, and the year France's capital became official?": (
        "(12+3)*2 equals 30, and Paris has been France's capital since 987 CE."
    ),
}


@observe(type="llm", model="fake-gpt")
async def plan_llm(prompt: str) -> list[str]:
    await asyncio.sleep(0.03)
    plan = _PLANS.get(prompt, [])
    update_current_span(input=prompt, output=str(plan))
    return plan


@observe(type="agent")
async def planner_agent(prompt: str) -> list[str]:
    plan = await plan_llm(prompt)
    update_current_span(input=prompt, output=str(plan))
    return plan


@observe(type="llm", model="fake-gpt")
async def rewrite_query_llm(prompt: str) -> str:
    await asyncio.sleep(0.02)
    rewritten = _REWRITES.get(prompt, prompt)
    update_current_span(input=prompt, output=rewritten)
    return rewritten


@observe(type="retriever", embedder="fake-embedder")
async def context_retriever(query: str) -> list[str]:
    await asyncio.sleep(0.02)
    lowered = query.lower()
    chunks = [v for k, v in _KNOWLEDGE_BASE.items() if k in lowered]
    update_current_span(input=query, output=chunks or ["<no context>"])
    return chunks


@observe(type="agent")
async def retriever_agent(prompt: str) -> list[str]:
    rewritten = await rewrite_query_llm(prompt)
    chunks = await context_retriever(rewritten)
    update_current_span(input=prompt, output=chunks)
    return chunks


@observe(type="tool")
async def calculator_tool(expression: str) -> str:
    await asyncio.sleep(0.02)
    result = _CALC_RESULTS.get(
        expression.replace(" ", ""),
        f"error: unknown expression {expression!r}",
    )
    update_current_span(input=expression, output=result)
    return result


@observe(type="llm", model="fake-gpt")
async def calc_interpret_llm(expression: str, raw: str) -> str:
    await asyncio.sleep(0.02)
    interpreted = f"{expression} = {raw}"
    update_current_span(input=f"{expression} -> {raw}", output=interpreted)
    return interpreted


@observe(type="agent")
async def calculator_agent(expression: str) -> str:
    raw = await calculator_tool(expression)
    interpreted = await calc_interpret_llm(expression, raw)
    update_current_span(input=expression, output=interpreted)
    return interpreted


@observe(type="tool")
async def knowledge_lookup_tool(key: str) -> str:
    await asyncio.sleep(0.02)
    result = _KNOWLEDGE_BASE.get(key, "<not found>")
    update_current_span(input=key, output=result)
    return result


@observe(type="llm", model="fake-gpt")
async def kb_summarize_llm(key: str, raw: str) -> str:
    await asyncio.sleep(0.02)
    summary = raw if "<not found>" not in raw else f"no info on {key!r}"
    update_current_span(input=raw, output=summary)
    return summary


@observe(type="agent")
async def knowledge_agent(key: str) -> str:
    raw = await knowledge_lookup_tool(key)
    summary = await kb_summarize_llm(key, raw)
    update_current_span(input=key, output=summary)
    return summary


@observe(type="llm", model="fake-gpt")
async def synthesize_llm(prompt: str, sub_results: list[str]) -> str:
    await asyncio.sleep(0.05)
    response = _SYNTHESIS.get(prompt, "I don't know.")
    update_current_span(
        input=f"results: {sub_results}\n\nquestion: {prompt}",
        output=response,
    )
    return response


@observe(type="agent")
async def synthesizer_agent(prompt: str, sub_results: list[str]) -> str:
    response = await synthesize_llm(prompt, sub_results)
    update_current_span(input=prompt, output=response)
    return response


@observe(type="agent")
async def orchestrator(prompt: str) -> str:
    update_current_trace(
        name="manual-evals-iterator",
        tags=["manual", "evals_iterator", "multi-agent"],
        metadata={"run_id": RUN_ID, "script": Path(__file__).stem},
        input=[{"role": "user", "content": prompt}],
    )

    plan = await planner_agent(prompt)
    context = await retriever_agent(prompt)

    sub_results: list[str] = list(context)
    for step in plan:
        kind, _, arg = step.partition(":")
        if kind == "calculator":
            sub_results.append(await calculator_agent(arg))
        elif kind == "knowledge":
            sub_results.append(await knowledge_agent(arg))

    response = await synthesizer_agent(prompt, sub_results)

    update_current_trace(output=response)
    update_current_span(input=prompt, output=response)
    return response


async def run_agent(prompt: str) -> str:
    with next_agent_span(metrics=[TaskCompletionMetric(threshold=0.2)]):
        return await orchestrator(prompt)


dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's 7 * 8?"),
        Golden(input="What's the capital of France?"),
        Golden(input="Name two primary colors."),
        Golden(
            input="What's (12+3)*2, and the year France's capital became official?"
        ),
    ]
)


for golden in dataset.evals_iterator(
    async_config=AsyncConfig(run_async=True),
    metrics=[TaskCompletionMetric(threshold=0.5)],
):
    task = asyncio.create_task(run_agent(golden.input))
    dataset.evaluate(task)
