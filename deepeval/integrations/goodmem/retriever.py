from dataclasses import dataclass, field
from typing import List, Optional

from deepeval.tracing import observe, update_retriever_span

from deepeval.integrations.goodmem.types import GoodMemChunk
from deepeval.integrations.goodmem.utils import goodmem_retrieve


@dataclass
class GoodMemConfig:
    """Configuration for connecting to a GoodMem instance.

    Supports both single-space and multi-space queries::

        # Single space (backward compatible)
        config = GoodMemConfig(base_url=..., api_key=..., space_id="abc")

        # Multiple spaces
        config = GoodMemConfig(base_url=..., api_key=..., space_ids=["abc", "def"])
    """

    base_url: str
    api_key: str
    space_ids: List[str] = field(default_factory=list)
    top_k: int = 5
    reranker: Optional[str] = None
    relevance_threshold: Optional[float] = None
    metadata_filter: Optional[str] = None
    embedder: Optional[str] = None

    # Backward compat: accept space_id= as a shorthand for a single space.
    space_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if self.space_id and not self.space_ids:
            self.space_ids = [self.space_id]
        if not self.space_ids:
            raise ValueError("Provide space_id or space_ids")


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

        # Plain text list for LLMTestCase.retrieval_context
        chunks = retriever.retrieve("What is machine learning?")

        # Structured chunks with scores and IDs
        detailed = retriever.retrieve_chunks("What is machine learning?")
    """

    def __init__(self, config: GoodMemConfig):
        self.config = config

    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant chunks from GoodMem for a query.

        Delegates to :meth:`retrieve_chunks` (which is traced) and
        extracts plain text.  Returns a list of text strings suitable
        for ``LLMTestCase.retrieval_context``.
        """
        chunks = self.retrieve_chunks(query)
        return [c.content for c in chunks if c.content]

    @observe(type="retriever", name="GoodMem Retriever")
    def retrieve_chunks(self, query: str) -> List[GoodMemChunk]:
        """Retrieve relevant chunks with full metadata.

        Returns ``GoodMemChunk`` objects containing content, relevance
        scores, chunk IDs, memory IDs, and space IDs.
        """
        update_retriever_span(
            embedder=self.config.embedder,
            top_k=self.config.top_k,
        )

        return goodmem_retrieve(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            space_ids=self.config.space_ids,
            query=query,
            top_k=self.config.top_k,
            reranker=self.config.reranker,
            relevance_threshold=self.config.relevance_threshold,
            metadata_filter=self.config.metadata_filter,
        )

    def retrieve_as_context(self, query: str) -> List[str]:
        """Alias for ``retrieve`` — returns chunks formatted for
        ``LLMTestCase.retrieval_context``."""
        return self.retrieve(query)
