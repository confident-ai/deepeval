from dataclasses import dataclass, field
from typing import List, Optional

from deepeval.tracing import observe, update_retriever_span

from deepeval.integrations.goodmem.utils import (
    goodmem_retrieve,
    parse_chunks_to_texts,
)


@dataclass
class GoodMemConfig:
    """Configuration for connecting to a GoodMem instance."""

    base_url: str
    api_key: str
    space_id: str
    top_k: int = 5
    reranker: Optional[str] = None
    relevance_threshold: Optional[float] = None
    metadata_filter: Optional[str] = None
    embedder: Optional[str] = None


class GoodMemRetriever:
    """DeepEval-integrated retriever for GoodMem.

    Wraps GoodMem's retrieval API with deepeval's ``@observe`` decorator
    so every retrieval call is automatically traced as a retriever span.

    Usage::

        from deepeval.integrations.goodmem import GoodMemRetriever, GoodMemConfig

        retriever = GoodMemRetriever(GoodMemConfig(
            base_url="https://api.goodmem.ai",
            api_key="sk-...",
            space_id="my-space",
        ))

        # Returns List[str] of chunk texts; traced in deepeval
        chunks = retriever.retrieve("What is machine learning?")
    """

    def __init__(self, config: GoodMemConfig):
        self.config = config

    @observe(type="retriever", name="GoodMem Retriever")
    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant chunks from GoodMem for a query.

        The call is automatically traced as a deepeval retriever span.
        Returns a list of text strings suitable for
        ``LLMTestCase.retrieval_context``.
        """
        update_retriever_span(
            embedder=self.config.embedder,
            top_k=self.config.top_k,
        )

        response = goodmem_retrieve(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            space_id=self.config.space_id,
            query=query,
            top_k=self.config.top_k,
            reranker=self.config.reranker,
            relevance_threshold=self.config.relevance_threshold,
            metadata_filter=self.config.metadata_filter,
        )

        return parse_chunks_to_texts(response)

    def retrieve_as_context(self, query: str) -> List[str]:
        """Alias for ``retrieve`` — returns chunks formatted for
        ``LLMTestCase.retrieval_context``."""
        return self.retrieve(query)
