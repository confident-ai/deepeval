"""
Retriever LangChain App: RAG with deterministic retriever
Complexity: MEDIUM - Tests retriever spans with ChatOpenAI

Uses a deterministic retriever that returns fixed documents,
combined with ChatOpenAI for response generation.
Uses RunnableLambda wrapper to ensure proper callback events for tracing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List


class DeterministicRetriever(BaseRetriever):
    """A retriever that returns fixed documents based on query keywords."""

    documents: dict = {
        "python": [
            Document(
                page_content="Python is a high-level programming language known for its simplicity.",
                metadata={"source": "doc1"},
            ),
            Document(
                page_content="Python supports multiple programming paradigms including procedural and OOP.",
                metadata={"source": "doc2"},
            ),
        ],
        "langchain": [
            Document(
                page_content="LangChain is a framework for developing applications powered by language models.",
                metadata={"source": "doc3"},
            ),
            Document(
                page_content="LangChain provides tools for chaining LLM calls and integrating with external data.",
                metadata={"source": "doc4"},
            ),
        ],
        "default": [
            Document(
                page_content="This is a general document about AI and machine learning.",
                metadata={"source": "doc5"},
            ),
            Document(
                page_content="Machine learning enables computers to learn from data without explicit programming.",
                metadata={"source": "doc6"},
            ),
        ],
    }

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents based on query keywords."""
        query_lower = query.lower()

        if "python" in query_lower:
            return self.documents["python"]
        elif "langchain" in query_lower:
            return self.documents["langchain"]
        else:
            return self.documents["default"]


# Shared retriever and LLM
retriever = DeterministicRetriever()
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, seed=42)


def _run_rag_chain(inputs: dict, config: RunnableConfig = None):
    """Run the RAG chain synchronously."""
    messages = inputs.get("messages", [])

    # Extract query from messages
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break
        elif isinstance(msg, tuple) and msg[0] == "human":
            query = msg[1]
            break

    # Retrieve documents
    docs = retriever.invoke(query, config=config)

    # Format context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create augmented prompt with system message for RAG
    augmented_messages = (
        [
            SystemMessage(
                content="You are a helpful assistant. Answer the user's question based ONLY on the provided context. Be concise and factual."
            )
        ]
        + list(messages)
        + [
            HumanMessage(
                content=f"Context:\n{context}\n\nAnswer based on the context above."
            )
        ]
    )

    # Generate response
    response = llm.invoke(augmented_messages, config=config)

    return {
        "messages": list(messages) + [response],
        "context": context,
        "source_documents": docs,
    }


async def _arun_rag_chain(inputs: dict, config: RunnableConfig = None):
    """Run the RAG chain asynchronously."""
    messages = inputs.get("messages", [])

    # Extract query from messages
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break
        elif isinstance(msg, tuple) and msg[0] == "human":
            query = msg[1]
            break

    # Retrieve documents
    docs = await retriever.ainvoke(query, config=config)

    # Format context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create augmented prompt with system message for RAG
    augmented_messages = (
        [
            SystemMessage(
                content="You are a helpful assistant. Answer the user's question based ONLY on the provided context. Be concise and factual."
            )
        ]
        + list(messages)
        + [
            HumanMessage(
                content=f"Context:\n{context}\n\nAnswer based on the context above."
            )
        ]
    )

    # Generate response
    response = await llm.ainvoke(augmented_messages, config=config)

    return {
        "messages": list(messages) + [response],
        "context": context,
        "source_documents": docs,
    }


# Wrap as RunnableLambda chains for proper callback event propagation
_rag_chain = RunnableLambda(_run_rag_chain).with_config(run_name="rag_chain")
_rag_async_chain = RunnableLambda(_arun_rag_chain).with_config(
    run_name="rag_chain"
)


def invoke_rag_app(inputs: dict, config: RunnableConfig = None):
    """Invoke the RAG app."""
    return _rag_chain.invoke(inputs, config=config)


async def ainvoke_rag_app(inputs: dict, config: RunnableConfig = None):
    """Async invoke the RAG app."""
    return await _rag_async_chain.ainvoke(inputs, config=config)
