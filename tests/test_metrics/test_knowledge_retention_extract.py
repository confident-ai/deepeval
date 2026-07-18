"""Unit tests for KnowledgeRetentionMetric knowledge extraction.

Verifies that the extract_json lambda in _generate_knowledges /
_a_generate_knowledges correctly constructs Knowledge objects from
the parsed LLM JSON without double-wrapping.
"""

import pytest
from pydantic import ValidationError

from deepeval.metrics.knowledge_retention.schema import Knowledge


class TestKnowledgeExtraction:
    """Regression tests for the double-wrap bug (issue #2512).

    The LLM returns JSON like ``{"data": {"Full Name": "Emily Chen"}}``.
    After ``trimAndLoadJson``, the result is a Python dict with a ``"data"``
    key.  The ``extract_json`` lambda must unpack this dict via
    ``Knowledge(**data)`` so that ``Knowledge.data`` receives the inner dict,
    not the outer one.
    """

    def test_knowledge_from_dict_with_data_key(self):
        """Knowledge(**data) should work when data has a 'data' key."""
        raw = {"data": {"Full Name": "Emily Chen"}}
        knowledge = Knowledge(**raw)
        assert knowledge.data == {"Full Name": "Emily Chen"}

    def test_knowledge_from_dict_with_list_values(self):
        """Knowledge should accept list-of-strings values."""
        raw = {
            "data": {
                "Dietary Restrictions": ["Vegetarian", "Peanut Allergy"],
                "Current Location": "Berlin",
            }
        }
        knowledge = Knowledge(**raw)
        assert knowledge.data["Dietary Restrictions"] == [
            "Vegetarian",
            "Peanut Allergy",
        ]
        assert knowledge.data["Current Location"] == "Berlin"

    def test_knowledge_from_empty_data(self):
        """An empty data dict should be accepted."""
        raw = {"data": {}}
        knowledge = Knowledge(**raw)
        assert knowledge.data == {}

    def test_knowledge_from_none_data(self):
        """None data should be accepted (field is Optional)."""
        raw = {"data": None}
        knowledge = Knowledge(**raw)
        assert knowledge.data is None

    def test_double_wrap_raises_validation_error(self):
        """Passing the outer dict as-is to data= must fail validation.

        This is the exact bug: ``Knowledge(data=data)`` where ``data``
        is ``{"data": {"Full Name": "Emily Chen"}}`` produces a nested
        dict whose inner value is another dict, violating the type
        ``Dict[str, Union[str, List[str]]]``.
        """
        raw = {"data": {"Full Name": "Emily Chen"}}
        with pytest.raises(ValidationError):
            Knowledge(data=raw)

    def test_knowledge_rejects_extra_fields(self):
        """ConfigDict(extra='forbid') should reject unexpected keys."""
        with pytest.raises(ValidationError):
            Knowledge(data={"Name": "Alice"}, unexpected_field="bad")
