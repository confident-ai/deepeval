"""
Retriever LangGraph App: RAG with deterministic retriever
Complexity: MEDIUM - Tests retriever spans with ChatOpenAI in LangGraph

Uses a deterministic retriever that returns fixed documents,
combined with ChatOpenAI for response generation in a LangGraph workflow.
"""

from typing import List, TypedDict

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun


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


class RAGState(TypedDict):
    """State for the RAG workflow."""

    messages: List[HumanMessage | AIMessage | SystemMessage]
    context: str
    source_documents: List[Document]


# Shared retriever and LLM
retriever = DeterministicRetriever()
retriever_with_metric_collection = DeterministicRetriever(
    metadata={"metric_collection": "retriever_quality"}
)
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, seed=42)


def retrieve_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """Retrieve documents based on the user's query."""
    messages = state.get("messages", [])

    # Extract query from messages
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    # Retrieve documents
    docs = retriever.invoke(query, config=config)

    # Format context
    context = "\n\n".join([doc.page_content for doc in docs])

    return {"context": context, "source_documents": docs}


def generate_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """Generate response based on retrieved context."""
    messages = state.get("messages", [])
    context = state.get("context", "")

    # Create augmented prompt with system message for RAG
    augmented_messages = [
        SystemMessage(
            content="You are a helpful assistant. Answer the user's question based ONLY on the provided context. Be concise and factual."
        ),
        *messages,
        HumanMessage(
            content=f"Context:\n{context}\n\nAnswer based on the context above."
        ),
    ]

    # Generate response
    response = llm.invoke(augmented_messages, config=config)

    return {"messages": [*messages, response]}


def retrieve_node_with_metric_collection(
    state: RAGState, config: RunnableConfig
) -> RAGState:
    """Retrieve documents using retriever with metric_collection metadata."""
    messages = state.get("messages", [])

    # Extract query from messages
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    # Retrieve documents using the metric_collection retriever
    docs = retriever_with_metric_collection.invoke(query, config=config)

    # Format context
    context = "\n\n".join([doc.page_content for doc in docs])

    return {"context": context, "source_documents": docs}


def build_app():
    """Build and compile the RAG workflow graph."""
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def build_app_with_metric_collection():
    """Build RAG workflow graph with retriever that has metric_collection."""
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node_with_metric_collection)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


app = build_app()
app_with_metric_collection = build_app_with_metric_collection()
