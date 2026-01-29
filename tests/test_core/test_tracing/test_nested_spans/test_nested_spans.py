import pytest
import asyncio
from deepeval.tracing import observe
from tests.test_core.test_tracing.conftest import trace_test


@observe()
def parent_function(data: str) -> str:
    result = child_function(data)
    return f"Parent: {result}"


@observe()
def child_function(data: str) -> str:
    return f"Child: {data}"


@observe(type="agent")
def agent_workflow(query: str) -> str:
    retrieved = retriever_step(query)
    response = llm_step(retrieved)
    return response


@observe(type="retriever", embedder="ada-002")
def retriever_step(query: str) -> str:
    return f"Retrieved docs for: {query}"


@observe(type="llm", model="gpt-4")
def llm_step(context: str) -> str:
    return f"Generated from: {context}"


@observe()
def deep_nesting_level_1(data: str) -> str:
    return deep_nesting_level_2(data)


@observe()
def deep_nesting_level_2(data: str) -> str:
    return deep_nesting_level_3(data)


@observe()
def deep_nesting_level_3(data: str) -> str:
    return f"Deep: {data}"


@observe()
def parent_with_multiple_children(data: str) -> str:
    result1 = first_child(data)
    result2 = second_child(data)
    result3 = third_child(data)
    return f"{result1} | {result2} | {result3}"


@observe()
def first_child(data: str) -> str:
    return f"First: {data}"


@observe()
def second_child(data: str) -> str:
    return f"Second: {data}"


@observe()
def third_child(data: str) -> str:
    return f"Third: {data}"


@observe(type="agent")
async def async_agent(query: str) -> str:
    docs = await async_retrieve(query)
    response = await async_generate(docs)
    return response


@observe(type="retriever")
async def async_retrieve(query: str) -> str:
    await asyncio.sleep(0.01)
    return f"Async docs: {query}"


@observe(type="llm", model="gpt-4")
async def async_generate(context: str) -> str:
    await asyncio.sleep(0.01)
    return f"Async response: {context}"


class TestNestedSpans:

    @trace_test("nested_spans/simple_nesting_schema.json")
    def test_simple_parent_child(self):
        parent_function("test")

    @trace_test("nested_spans/agent_workflow_schema.json")
    def test_agent_workflow_nesting(self):
        agent_workflow("search query")

    @trace_test("nested_spans/deep_nesting_schema.json")
    def test_deep_nesting(self):
        deep_nesting_level_1("data")

    @trace_test("nested_spans/multiple_children_schema.json")
    def test_multiple_children(self):
        parent_with_multiple_children("data")

    @trace_test("nested_spans/async_nesting_schema.json")
    @pytest.mark.asyncio
    async def test_async_nesting(self):
        await async_agent("async query")
