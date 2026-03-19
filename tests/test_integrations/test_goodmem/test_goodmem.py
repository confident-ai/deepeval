import json
from unittest.mock import patch, MagicMock

import pytest

from deepeval.integrations.goodmem import GoodMemRetriever, GoodMemConfig
from deepeval.integrations.goodmem.utils import (
    goodmem_retrieve,
    _parse_ndjson_response,
    parse_chunks_to_texts,
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
        assert len(result["chunks"]) == 2
        assert (
            result["chunks"][0]["content"]
            == "Python is a programming language."
        )
        assert (
            result["chunks"][1]["content"]
            == "Python was created by Guido van Rossum."
        )

    def test_extracts_chunk_ids(self):
        result = _parse_ndjson_response(SAMPLE_NDJSON_RESPONSE)
        assert result["chunks"][0]["chunk_id"] == "chunk-1"
        assert result["chunks"][1]["chunk_id"] == "chunk-2"

    def test_extracts_memory_ids(self):
        result = _parse_ndjson_response(SAMPLE_NDJSON_RESPONSE)
        assert result["chunks"][0]["memory_id"] == "mem-1"

    def test_extracts_relevance_scores(self):
        result = _parse_ndjson_response(SAMPLE_NDJSON_RESPONSE)
        assert result["chunks"][0]["relevance_score"] == -0.25

    def test_empty_response(self):
        result = _parse_ndjson_response("")
        assert result["chunks"] == []

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
        assert len(result["chunks"]) == 1


class TestParseChunksToTexts:
    def test_returns_text_list(self):
        response = {
            "chunks": [
                {"content": "chunk one"},
                {"content": "chunk two"},
            ]
        }
        assert parse_chunks_to_texts(response) == ["chunk one", "chunk two"]

    def test_skips_empty_content(self):
        response = {
            "chunks": [
                {"content": ""},
                {"content": "valid"},
            ]
        }
        assert parse_chunks_to_texts(response) == ["valid"]


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
            space_id="space-123",
            query="What is Python?",
            top_k=3,
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["message"] == "What is Python?"
        assert call_kwargs[1]["json"]["spaceKeys"] == [{"spaceId": "space-123"}]
        assert call_kwargs[1]["json"]["requestedSize"] == 3
        assert call_kwargs[1]["headers"]["x-api-key"] == "test-key"
        assert len(result["chunks"]) == 2

    @patch("deepeval.integrations.goodmem.utils.requests.post")
    def test_includes_metadata_filter(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = SAMPLE_NDJSON_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        goodmem_retrieve(
            base_url="https://api.goodmem.ai",
            api_key="test-key",
            space_id="space-123",
            query="test",
            metadata_filter="source = 'wiki'",
        )

        body = mock_post.call_args[1]["json"]
        assert body["spaceKeys"][0]["filter"] == "source = 'wiki'"


# ---------------------------------------------------------------------------
# GoodMemRetriever tests
# ---------------------------------------------------------------------------


class TestGoodMemRetriever:
    @patch("deepeval.integrations.goodmem.retriever.goodmem_retrieve")
    def test_retrieve_returns_texts(self, mock_retrieve, retriever):
        mock_retrieve.return_value = {
            "chunks": [
                {"content": "chunk one"},
                {"content": "chunk two"},
            ]
        }

        result = retriever.retrieve("test query")
        assert result == ["chunk one", "chunk two"]
        mock_retrieve.assert_called_once_with(
            base_url="https://api.goodmem.ai",
            api_key="test-key",
            space_id="space-123",
            query="test query",
            top_k=3,
            reranker=None,
            relevance_threshold=None,
            metadata_filter=None,
        )

    @patch("deepeval.integrations.goodmem.retriever.goodmem_retrieve")
    def test_retrieve_as_context_delegates(self, mock_retrieve, retriever):
        mock_retrieve.return_value = {"chunks": [{"content": "ctx"}]}
        result = retriever.retrieve_as_context("query")
        assert result == ["ctx"]

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
