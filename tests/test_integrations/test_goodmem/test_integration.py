"""
Integration test: GoodMem retrieval → OpenAI generation → DeepEval evaluation.

Requires a running GoodMem instance with a populated space and an OpenAI key.
Skipped automatically unless all required env vars are set.

Run with::

    pytest tests/test_integrations/test_goodmem/test_integration.py -m integration -v

Required env vars:
    GOODMEM_BASE_URL   – e.g. https://api.goodmem.ai
    GOODMEM_API_KEY    – GoodMem API key
    GOODMEM_SPACE_ID   – Space ID containing retrievable content
    OPENAI_API_KEY     – OpenAI API key for generation and metrics
"""

import os

import pytest

REQUIRED_VARS = [
    "GOODMEM_BASE_URL",
    "GOODMEM_API_KEY",
    "GOODMEM_SPACE_ID",
    "OPENAI_API_KEY",
]
_missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        len(_missing) > 0,
        reason=f"Missing env vars: {', '.join(_missing)}",
    ),
]


@pytest.fixture(scope="module")
def retriever():
    from deepeval.integrations.goodmem import GoodMemConfig, GoodMemRetriever

    return GoodMemRetriever(
        GoodMemConfig(
            base_url=os.environ.get("GOODMEM_BASE_URL", ""),
            api_key=os.environ.get("GOODMEM_API_KEY", ""),
            space_id=os.environ.get("GOODMEM_SPACE_ID", ""),
            top_k=3,
        )
    )


@pytest.fixture(scope="module")
def openai_client():
    from openai import OpenAI

    return OpenAI()


SYSTEM_PROMPT = (
    "Answer the question accurately based only on the provided context. "
    "If the context doesn't contain enough information, say so."
)
GENERATION_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------


class TestRetrieve:
    """Verify that live retrieval returns usable results."""

    def test_retrieve_returns_strings(self, retriever):
        results = retriever.retrieve("What is energy?")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, str) for r in results)

    def test_retrieve_chunks_returns_structured(self, retriever):
        from deepeval.integrations.goodmem import GoodMemChunk

        chunks = retriever.retrieve_chunks("What is energy?")
        assert len(chunks) > 0
        assert all(isinstance(c, GoodMemChunk) for c in chunks)

    def test_chunk_has_metadata(self, retriever):
        chunks = retriever.retrieve_chunks("What is energy?")
        chunk = chunks[0]
        assert chunk.content
        assert chunk.score is not None
        assert chunk.chunk_id
        assert chunk.memory_id

    def test_top_k_respected(self, retriever):
        chunks = retriever.retrieve_chunks("test query")
        assert len(chunks) <= retriever.config.top_k


# ---------------------------------------------------------------------------
# RAG pipeline test
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    """End-to-end: retrieve → generate → evaluate with DeepEval metrics."""

    @staticmethod
    def _generate(client, chunks, query):
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{chr(10).join(chunks)}\n\nQuestion: {query}",
                },
            ],
        )
        return response.choices[0].message.content

    def test_rag_answer_relevancy(self, retriever, openai_client):
        """Retrieve, generate, and verify answer relevancy score."""
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        query = "What are the main forms of energy in physics?"
        chunks = retriever.retrieve(query)
        answer = self._generate(openai_client, chunks, query)

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=chunks,
        )
        metric = AnswerRelevancyMetric(model="gpt-4o-mini")
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.score >= 0.0

    def test_rag_contextual_relevancy(self, retriever, openai_client):
        """Retrieve, generate, and verify contextual relevancy score."""
        from deepeval.metrics import ContextualRelevancyMetric
        from deepeval.test_case import LLMTestCase

        query = "What are the main forms of energy in physics?"
        chunks = retriever.retrieve(query)
        answer = self._generate(openai_client, chunks, query)

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=chunks,
        )
        metric = ContextualRelevancyMetric(model="gpt-4o-mini")
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.score >= 0.0

    def test_batch_evaluate(self, retriever, openai_client):
        """Build multiple test cases and run batch evaluation."""
        from deepeval import evaluate
        from deepeval.evaluate import AsyncConfig
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        queries = [
            "What are the main forms of energy in physics?",
            "Who created American Idol and when did it first air?",
        ]

        test_cases = []
        for query in queries:
            chunks = retriever.retrieve(query)
            answer = self._generate(openai_client, chunks, query)
            test_cases.append(
                LLMTestCase(
                    input=query,
                    actual_output=answer,
                    retrieval_context=chunks,
                )
            )

        results = evaluate(
            test_cases,
            [AnswerRelevancyMetric(model="gpt-4o-mini")],
            async_config=AsyncConfig(max_concurrent=2, throttle_value=1),
        )
        assert results is not None
