"""
Simple LlamaIndex App
Complexity: LOW - Basic Query Engine with no tools or retrieval
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI

# Deterministic LLM
llm = OpenAI(model="gpt-4o", temperature=0.0)


def get_simple_engine():
    """
    Returns a basic query engine over a single mock document.
    """
    node = TextNode(
        text="LlamaIndex is a data framework for LLM applications.",
        id_="fixed_simple_node_id",
    )

    # Initialize index directly from the list of nodes
    index = VectorStoreIndex(nodes=[node])

    return index.as_query_engine(llm=llm)
