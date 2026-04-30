from deepeval.tracing import observe, update_retriever_span
from tests.test_core.test_tracing.conftest import trace_test


@observe(type="retriever", embedder="text-embedding-ada-002")
def retrieve_documents(query: str, top_k: int = 5) -> list:
    documents = [f"Document {i} about {query}" for i in range(top_k)]
    update_retriever_span(
        top_k=top_k,
        chunk_size=512,
    )
    return documents


@observe(type="retriever", embedder="all-MiniLM-L6-v2")
def retrieve_with_custom_embedder(query: str) -> list:
    docs = [f"Result for: {query}"]
    update_retriever_span(
        top_k=3,
        chunk_size=256,
    )
    return docs


@observe(type="retriever")
def retrieve_minimal(query: str) -> list:
    return [f"Result: {query}"]


@observe(type="retriever", embedder="ada-002", name="custom_retriever_name")
def retriever_with_custom_name(query: str) -> list:
    return [f"Named retriever: {query}"]


@observe(type="retriever")
def retriever_full_attributes(query: str) -> list:
    results = ["Chunk 1", "Chunk 2", "Chunk 3"]
    update_retriever_span(
        embedder="voyage-code-2",
        top_k=3,
        chunk_size=1024,
    )
    return results


@observe(type="retriever", embedder="initial-embedder")
def retriever_override_embedder(query: str) -> list:
    results = ["Result"]
    update_retriever_span(embedder="new-embedder")
    return results


class TestRetrieverSpan:

    @trace_test("span_types/retriever_span_schema.json")
    def test_retriever_with_embedder(self):
        retrieve_documents("AI research", top_k=3)

    @trace_test("span_types/retriever_custom_embedder_schema.json")
    def test_retriever_custom_embedder(self):
        retrieve_with_custom_embedder("machine learning")

    @trace_test("span_types/retriever_minimal_schema.json")
    def test_retriever_minimal(self):
        retrieve_minimal("search query")

    @trace_test("span_types/retriever_custom_name_schema.json")
    def test_retriever_with_custom_name(self):
        retriever_with_custom_name("test query")

    @trace_test("span_types/retriever_full_attributes_schema.json")
    def test_retriever_full_attributes(self):
        retriever_full_attributes("machine learning")

    @trace_test("span_types/retriever_override_embedder_schema.json")
    def test_retriever_override_embedder(self):
        retriever_override_embedder("test")
