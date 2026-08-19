"""Unit tests for the shared OpenRouter helpers. No SDK or network required."""

from types import SimpleNamespace

import pytest

from deepeval.model_integrations.utils import (
    detect_provider_from_base_url,
    extract_openrouter_metadata,
)


class FakeURL:
    """Stands in for `httpx.URL`, which exposes the host as an attribute."""

    def __init__(self, host):
        self.host = host


@pytest.mark.parametrize(
    "base_url",
    [
        "https://openrouter.ai/api/v1",
        "https://OpenRouter.ai/api/v1",
        FakeURL("openrouter.ai"),
        # Suffix match, so regional/vanity subdomains still resolve.
        FakeURL("eu.openrouter.ai"),
    ],
)
def test_detects_openrouter(base_url):
    assert detect_provider_from_base_url(base_url) == "OpenRouter"


@pytest.mark.parametrize(
    "base_url",
    [
        None,
        "",
        "https://api.openai.com/v1",
        FakeURL("my-proxy.internal"),
        # Must not match a lookalike domain that merely contains the name.
        FakeURL("openrouter.ai.evil.com"),
    ],
)
def test_does_not_detect_other_hosts(base_url):
    assert detect_provider_from_base_url(base_url) is None


def test_extracts_metadata_from_chat_completions_shape():
    response = SimpleNamespace(
        id="gen-123",
        provider="Anthropic",
        usage=SimpleNamespace(
            cost=0.5,
            is_byok=True,
            cost_details=None,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=4, cache_write_tokens=9
            ),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )

    metadata = extract_openrouter_metadata(response)

    assert metadata["generation_id"] == "gen-123"
    assert metadata["upstream_provider"] == "Anthropic"
    assert metadata["cost"] == 0.5
    assert metadata["is_byok"] is True
    assert metadata["cached_tokens"] == 4
    assert metadata["cache_write_tokens"] == 9
    assert metadata["reasoning_tokens"] == 2


def test_extracts_metadata_from_responses_api_shape():
    """The Responses API names the same details input/output rather than
    prompt/completion."""
    response = SimpleNamespace(
        id="gen-456",
        usage=SimpleNamespace(
            cost=0.25,
            input_tokens_details=SimpleNamespace(cached_tokens=7),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
    )

    metadata = extract_openrouter_metadata(response)

    assert metadata["generation_id"] == "gen-456"
    assert metadata["cost"] == 0.25
    assert metadata["cached_tokens"] == 7
    assert metadata["reasoning_tokens"] == 3


def test_returns_none_when_nothing_openrouter_specific_is_present():
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    )
    assert extract_openrouter_metadata(response) is None


def test_unset_sentinels_are_treated_as_absent():
    """The openrouter SDK marks absent nullable fields with an UNSET sentinel
    that is not None, so it would otherwise serialize as the string 'Unset()'.
    """

    class Unset:
        def __repr__(self):
            return "Unset()"

    response = SimpleNamespace(
        id="gen-789",
        usage=SimpleNamespace(cost=1.0, cost_details=Unset(), is_byok=Unset()),
    )

    metadata = extract_openrouter_metadata(response)

    assert metadata == {"generation_id": "gen-789", "cost": 1.0}


def test_never_raises_on_a_hostile_response():
    class Exploding:
        @property
        def id(self):
            raise RuntimeError("boom")

    assert extract_openrouter_metadata(Exploding()) is None
