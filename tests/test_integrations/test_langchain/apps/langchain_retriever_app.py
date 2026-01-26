"""
Retriever LangChain App: RAG with deterministic retriever
Complexity: MEDIUM - Tests retriever spans with deterministic results

Uses a fake retriever that returns fixed documents.
"""

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
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


def _create_rag_chain(response_text: str):
    """Create a RAG chain with the specified response."""
    retriever = DeterministicRetriever()

    def run_rag(inputs: dict, config: RunnableConfig = None):
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

        # Create LLM with appropriate response based on query
        if "python" in query.lower():
            final_response = "Based on the retrieved documents, Python is a high-level programming language known for its simplicity and support for multiple paradigms."
        elif "langchain" in query.lower():
            final_response = "According to the documents, LangChain is a framework for building LLM-powered applications with tools for chaining calls and data integration."
        else:
            final_response = "Based on the context, AI and machine learning enable computers to learn patterns from data."

        llm = FakeMessagesListChatModel(
            responses=[AIMessage(content=final_response)]
        )

        # Create augmented prompt
        augmented_messages = list(messages) + [
            HumanMessage(
                content=f"Context:\n{context}\n\nAnswer based on the context above."
            )
        ]

        # Generate response
        response = llm.invoke(augmented_messages, config=config)

        return {
            "messages": list(messages) + [response],
            "context": context,
            "source_documents": docs,
        }

    return RunnableLambda(run_rag)


chain = _create_rag_chain("default")


def invoke_rag_app(inputs: dict, config: RunnableConfig = None):
    """Invoke the RAG app."""
    return chain.invoke(inputs, config=config)


async def ainvoke_rag_app(inputs: dict, config: RunnableConfig = None):
    """Async invoke the RAG app."""
    return await chain.ainvoke(inputs, config=config)
