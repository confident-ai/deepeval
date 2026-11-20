"""A provider that accepts `schema` must not have it silently dropped on retry."""

import pytest

from deepeval.models.base_model import DeepEvalBaseLLM


class _AcceptsSchemaButFails(DeepEvalBaseLLM):
    """Takes `schema`, but raises TypeError inside its own code."""

    def __init__(self):
        self.calls: list[dict] = []

    def load_model(self):
        return self

    def generate(self, *args, **kwargs):
        self.calls.append(sorted(kwargs))
        if "schema" in kwargs:
            raise TypeError("'NoneType' object is not subscriptable")
        return "plain text (schema dropped)"

    async def a_generate(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def get_model_name(self):
        return "accepts-schema-but-fails"


class _NoSchemaSupport(DeepEvalBaseLLM):
    """Genuinely does not accept a schema argument."""

    def load_model(self):
        return self

    def generate(self, prompt):
        return f"plain:{prompt}"

    async def a_generate(self, prompt):
        return self.generate(prompt)

    def get_model_name(self):
        return "no-schema"


def test_internal_type_error_is_not_treated_as_missing_schema_support():
    """The fallback exists for providers without the kwarg, not for their bugs.

    Catching TypeError cannot tell the two apart, so an accepted schema used to
    be dropped silently: the provider was simply called again without it.
    """
    model = _AcceptsSchemaButFails()

    with pytest.raises(TypeError):
        model.generate_with_schema("prompt", schema=dict)

    assert model.calls == [["schema"]]


def test_provider_without_schema_support_still_falls_back():
    model = _NoSchemaSupport()

    assert model.generate_with_schema("prompt", schema=dict) == "plain:prompt"
