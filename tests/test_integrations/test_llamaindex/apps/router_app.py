"""
Router LlamaIndex App
Complexity: HIGH - Routing between different engines
"""
from llama_index.core import Settings
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.core.base.response.schema import Response
from tests.test_integrations.test_llamaindex.apps.rag_app import get_rag_engine


rag_engine = get_rag_engine()


class MockMathEngine:
    """A fake query engine that just handles math queries string-wise."""

    def query(self, query_str: str):

        return Response(response="Calculated Result: 42 (Mock)")

    async def aquery(self, query_str: str):

        return Response(response="Calculated Result: 42 (Mock)")


def get_router_engine():
    """Builds a router that selects between RAG and Math."""
    Settings.llm = OpenAI(model="gpt-4o", temperature=0.0)

    rag_tool = QueryEngineTool.from_defaults(
        query_engine=rag_engine,
        description="Useful for questions about Python or LlamaIndex programming.",
    )

    math_tool = QueryEngineTool.from_defaults(
        query_engine=MockMathEngine(),
        description="Useful for questions about math or calculations.",
    )

    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=Settings.llm),
        query_engine_tools=[rag_tool, math_tool],
    )