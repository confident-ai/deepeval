"""Tests for serialize_retrieval_context / reconstruct_retrieval_context round-trip.

Specifically covers the case where RetrievedContextData.context contains
newline characters, which previously caused _RETRIEVED_CONTEXT_MARKER to fail
to match (the regex used `.` without re.DOTALL, so it could not cross `\n`),
silently returning the raw serialized string instead of the reconstructed object.
"""

import pytest
from deepeval.dataset.utils import (
    serialize_retrieval_context,
    reconstruct_retrieval_context,
)
from deepeval.test_case import RetrievedContextData


class TestRetrievalContextRoundTrip:
    def test_plain_string_roundtrip(self):
        ctx = ["plain context string"]
        assert (
            reconstruct_retrieval_context(serialize_retrieval_context(ctx))
            == ctx
        )

    def test_retrieved_context_data_simple_roundtrip(self):
        item = RetrievedContextData(source="doc1", context="some context")
        ctx = [item]
        result = reconstruct_retrieval_context(serialize_retrieval_context(ctx))
        assert len(result) == 1
        assert isinstance(result[0], RetrievedContextData)
        assert result[0].source == "doc1"
        assert result[0].context == "some context"

    def test_retrieved_context_data_newline_in_context(self):
        """Regression: context containing \\n must survive a serialize/reconstruct round-trip.

        Before the fix, _RETRIEVED_CONTEXT_MARKER used `.` without re.DOTALL,
        so the match returned None for any context that contained a newline.
        reconstruct_retrieval_context then silently passed through the raw
        serialized string instead of rebuilding the RetrievedContextData object.
        """
        item = RetrievedContextData(
            source="doc1", context="first line\nsecond line"
        )
        ctx = [item]
        serialized = serialize_retrieval_context(ctx)
        result = reconstruct_retrieval_context(serialized)
        assert len(result) == 1, "Expected one item back"
        assert isinstance(
            result[0], RetrievedContextData
        ), f"Expected RetrievedContextData, got {type(result[0])}: {result[0]!r}"
        assert result[0].source == "doc1"
        assert result[0].context == "first line\nsecond line"

    def test_retrieved_context_data_newline_in_source(self):
        """Source containing \\n must also survive the round-trip."""
        item = RetrievedContextData(
            source="section1\nsection2", context="normal context"
        )
        ctx = [item]
        result = reconstruct_retrieval_context(serialize_retrieval_context(ctx))
        assert isinstance(result[0], RetrievedContextData)
        assert result[0].source == "section1\nsection2"
        assert result[0].context == "normal context"

    def test_mixed_list_roundtrip(self):
        """Mixed plain strings and RetrievedContextData in one list."""
        items = [
            "plain",
            RetrievedContextData(source="s1", context="line1\nline2"),
            "another plain",
        ]
        result = reconstruct_retrieval_context(
            serialize_retrieval_context(items)
        )
        assert result[0] == "plain"
        assert isinstance(result[1], RetrievedContextData)
        assert result[1].context == "line1\nline2"
        assert result[2] == "another plain"

    def test_none_passthrough(self):
        assert serialize_retrieval_context(None) is None
        assert reconstruct_retrieval_context(None) is None

    def test_source_with_comma_roundtrip(self):
        """Source containing a comma should still round-trip correctly."""
        item = RetrievedContextData(source="foo,bar", context="ctx")
        result = reconstruct_retrieval_context(
            serialize_retrieval_context([item])
        )
        assert isinstance(result[0], RetrievedContextData)
        assert result[0].source == "foo,bar"
        assert result[0].context == "ctx"
