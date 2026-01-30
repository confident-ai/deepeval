"""
RAG LlamaIndex App
Complexity: MEDIUM - Custom Retriever + Synthesizer
"""

from typing import List
from llama_index.core import QueryBundle, get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

# Deterministic LLM
llm = OpenAI(model="gpt-4o", temperature=0.0)


class DeterministicRetriever(BaseRetriever):
    """
    A retriever that returns fixed nodes based on key terms.
    """

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str.lower()
        nodes = []

        if "python" in query_str:
            nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text="Python is a high-level, interpreted programming language known for its simplicity.",
                        id_="fixed_node_python",
                    ),
                    score=0.95,
                )
            )
        elif "llama" in query_str:
            nodes.append(
                NodeWithScore(
                    node=TextNode(
                        text="LlamaIndex is a data framework for your LLM applications.",
                        id_="fixed_node_llama",
                    ),
                    score=0.98,
                )
            )

        return nodes


def get_rag_engine():
    """Builds and returns the deterministic RAG query engine."""
    retriever = DeterministicRetriever()
    response_synthesizer = get_response_synthesizer(llm=llm)

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
