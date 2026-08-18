from types import SimpleNamespace

import pytest

from deepeval.integrations.agentcore.instrumentator import (
    AgentCoreSpanInterceptor,
)
from deepeval.integrations.strands.instrumentator import StrandsSpanInterceptor


class _Span:
    def __init__(self, attributes):
        self.attributes = attributes
        self._attributes = attributes
        self.events = []
        self.name = "chat"

    def set_attribute(self, key, value):
        self.attributes[key] = value


@pytest.mark.parametrize(
    "interceptor_class",
    [AgentCoreSpanInterceptor, StrandsSpanInterceptor],
)
@pytest.mark.parametrize(
    ("modern_key", "legacy_key", "target_key"),
    [
        (
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.prompt_tokens",
            "confident.llm.input_token_count",
        ),
        (
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.completion_tokens",
            "confident.llm.output_token_count",
        ),
    ],
)
@pytest.mark.parametrize(
    ("usage_attributes", "expected"),
    [
        ({"modern": 0, "legacy": 17}, 0),
        ({"legacy": 17}, 17),
    ],
)
def test_genai_token_usage_prefers_present_modern_attribute(
    interceptor_class,
    modern_key,
    legacy_key,
    target_key,
    usage_attributes,
    expected,
):
    attributes = {"gen_ai.operation.name": "chat"}
    if "modern" in usage_attributes:
        attributes[modern_key] = usage_attributes["modern"]
    if "legacy" in usage_attributes:
        attributes[legacy_key] = usage_attributes["legacy"]

    span = _Span(attributes)
    interceptor = interceptor_class(SimpleNamespace())

    interceptor._serialize_framework_attrs(span)

    assert span.attributes[target_key] == expected
