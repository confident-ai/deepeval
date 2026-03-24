import json
from unittest.mock import patch, MagicMock

import pytest

from deepeval.integrations.goodmem import (
    GoodMemRetriever,
    GoodMemConfig,
    GoodMemChunk,
)
from deepeval.integrations.goodmem.utils import (
    goodmem_retrieve,
    _parse_ndjson_response,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_NDJSON_RESPONSE = "\n".join(
    [
        json.dumps(
            {
                "resultSetBoundary": {
                    "resultSetId": "abc-123",
                    "kind": "BEGIN",
                    "stageName": "retrieve",
                    "expectedItems": 2,
                }
            }
        ),
        json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "chunk-1",
                            "memoryId": "mem-1",
                            "chunkText": "Python is a programming language.",
                        },
                        "relevanceScore": -0.25,
                    }
                }
            }
        ),
        json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "chunk-2",
                            "memoryId": "mem-2",
                            "chunkText": "Python was created by Guido van Rossum.",
                        },
                        "relevanceScore": -0.40,
                    }
                }
            }
        ),
        json.dumps(
            {
                "resultSetBoundary": {
                    "resultSetId": "abc-123",
                    "kind": "END",
                    "stageName": "",
                }
            }
        ),
    ]
)


@pytest.fixture
def config():
    return GoodMemConfig(
        base_url="https://api.goodmem.ai",
        api_key="test-key",
        space_id="space-123",
        top_k=3,
        embedder="text-embedding-3-small",
    )


@pytest.fixture
def retriever(config):
    return GoodMemRetriever(config)


# ---------------------------------------------------------------------------
# utils tests
# ---------------------------------------------------------------------------


class TestParseNdjsonResponse:
    def test_parses_chunks(self):
        result = _parse_ndjson_response(SAMPLE_NDJSON_RESPONSE)
        assert len(result) == 2
        assert isinstance(result[0], GoodMemChunk)
        assert result[0].content == "Python is a programming language."
        assert result[1].content == "Python was created by Guido van Rossum."

    def test_extracts_chunk_ids(self):
        result = _parse_ndjson_response(SAMPLE_NDJSON_RESPONSE)
        assert result[0].chunk_id == "chunk-1"
        assert result[1].chunk_id == "chunk-2"

    def test_extracts_memory_ids(self):
        result = _parse_ndjson_response(SAMPLE_NDJSON_RESPONSE)
        assert result[0].memory_id == "mem-1"

    def test_extracts_relevance_scores(self):
        result = _parse_ndjson_response(SAMPLE_NDJSON_RESPONSE)
        assert result[0].score == -0.25

    def test_empty_response(self):
        result = _parse_ndjson_response("")
        assert result == []

    def test_ignores_malformed_lines(self):
        text = "not json\n" + json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "c1",
                            "memoryId": "m1",
                            "chunkText": "valid",
                        },
                        "relevanceScore": -0.1,
                    }
                }
            }
        )
        result = _parse_ndjson_response(text)
        assert len(result) == 1
        assert result[0].content == "valid"


# ---------------------------------------------------------------------------
# goodmem_retrieve tests
# ---------------------------------------------------------------------------


class TestGoodmemRetrieve:
    @patch("deepeval.integrations.goodmem.utils.requests.post")
    def test_sends_correct_request(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = SAMPLE_NDJSON_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = goodmem_retrieve(
            base_url="https://api.goodmem.ai",
            api_key="test-key",
            space_ids=["space-123"],
            query="What is Python?",
            top_k=3,
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["message"] == "What is Python?"
        assert call_kwargs[1]["json"]["spaceKeys"] == [
            {"spaceId": "space-123"}
        ]
        assert call_kwargs[1]["json"]["requestedSize"] == 3
        assert call_kwargs[1]["headers"]["x-api-key"] == "test-key"
        assert len(result) == 2
        assert isinstance(result[0], GoodMemChunk)

    @patch("deepeval.integrations.goodmem.utils.requests.post")
    def test_includes_metadata_filter(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = SAMPLE_NDJSON_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        goodmem_retrieve(
            base_url="https://api.goodmem.ai",
            api_key="test-key",
            space_ids=["space-123"],
            query="test",
            metadata_filter="source = 'wiki'",
        )

        body = mock_post.call_args[1]["json"]
        assert body["spaceKeys"][0]["filter"] == "source = 'wiki'"

    @patch("deepeval.integrations.goodmem.utils.requests.post")
    def test_multi_space_request(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = SAMPLE_NDJSON_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        goodmem_retrieve(
            base_url="https://api.goodmem.ai",
            api_key="test-key",
            space_ids=["space-a", "space-b"],
            query="test",
        )

        body = mock_post.call_args[1]["json"]
        assert len(body["spaceKeys"]) == 2
        assert body["spaceKeys"][0] == {"spaceId": "space-a"}
        assert body["spaceKeys"][1] == {"spaceId": "space-b"}


# ---------------------------------------------------------------------------
# GoodMemRetriever tests
# ---------------------------------------------------------------------------


class TestGoodMemRetriever:
    @patch("deepeval.integrations.goodmem.utils.requests.post")
    def test_retrieve_returns_texts(self, mock_post, retriever):
        mock_response = MagicMock()
        mock_response.text = SAMPLE_NDJSON_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = retriever.retrieve("test query")
        assert result == [
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
        ]

    @patch("deepeval.integrations.goodmem.utils.requests.post")
    def test_retrieve_chunks_returns_structured(self, mock_post, retriever):
        mock_response = MagicMock()
        mock_response.text = SAMPLE_NDJSON_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = retriever.retrieve_chunks("test query")
        assert len(result) == 2
        assert isinstance(result[0], GoodMemChunk)
        assert result[0].content == "Python is a programming language."
        assert result[0].score == -0.25
        assert result[0].chunk_id == "chunk-1"
        assert result[0].memory_id == "mem-1"

    @patch("deepeval.integrations.goodmem.utils.requests.post")
    def test_retrieve_as_context_delegates(self, mock_post, retriever):
        mock_response = MagicMock()
        mock_response.text = SAMPLE_NDJSON_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = retriever.retrieve_as_context("query")
        assert result == [
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
        ]

    def test_config_defaults(self):
        config = GoodMemConfig(
            base_url="https://api.goodmem.ai",
            api_key="key",
            space_id="space",
        )
        assert config.top_k == 5
        assert config.reranker is None
        assert config.relevance_threshold is None
        assert config.metadata_filter is None
        assert config.embedder is None

    def test_config_space_id_backward_compat(self):
        config = GoodMemConfig(
            base_url="https://api.goodmem.ai",
            api_key="key",
            space_id="single-space",
        )
        assert config.space_ids == ["single-space"]

    def test_config_multi_space(self):
        config = GoodMemConfig(
            base_url="https://api.goodmem.ai",
            api_key="key",
            space_ids=["space-a", "space-b"],
        )
        assert config.space_ids == ["space-a", "space-b"]

    def test_config_requires_space(self):
        with pytest.raises(ValueError, match="space_id or space_ids"):
            GoodMemConfig(
                base_url="https://api.goodmem.ai",
                api_key="key",
            )
