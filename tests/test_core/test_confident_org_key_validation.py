import pytest
from deepeval.confident.api import Api, _is_organization_key
from deepeval.confident.types import ConfidentApiError


def test_is_organization_key_detects_org_scope():
    assert _is_organization_key("confident_us_org_abc123")
    assert _is_organization_key("confident_eu_org_abc123")
    assert _is_organization_key("CONFIDENT_US_ORG_ABC")  # case insensitive
    assert _is_organization_key("  confident_us_org_abc123  ")  # stripped


def test_is_organization_key_allows_project_keys():
    assert not _is_organization_key("confident_us_proj_abc123")
    assert not _is_organization_key("confident_eu_proj_abc123")
    assert not _is_organization_key("confident_us_global_abc")
    assert not _is_organization_key("random_key")
    assert not _is_organization_key("")


def test_api_rejects_organization_key():
    with pytest.raises(
        ConfidentApiError, match="organization API keys cannot be used"
    ):
        Api(api_key="confident_us_org_test123")

    with pytest.raises(
        ConfidentApiError, match="organization API keys cannot be used"
    ):
        Api(api_key="confident_eu_org_test123")


def test_api_rejects_org_key_with_whitespace():
    with pytest.raises(
        ConfidentApiError, match="organization API keys cannot be used"
    ):
        Api(api_key="  confident_us_org_test123  ")


def test_api_accepts_project_key(monkeypatch):
    # Use project key - should not raise org error (may fail on network later, but init should succeed)
    # Mock get_base_api_url to avoid network
    monkeypatch.setattr(
        "deepeval.confident.api.get_base_api_url",
        lambda: "https://api.confident-ai.com",
    )
    api = Api(api_key="confident_us_proj_test123")
    assert api.api_key == "confident_us_proj_test123"


def test_api_strips_whitespace_from_key(monkeypatch):
    monkeypatch.setattr(
        "deepeval.confident.api.get_base_api_url",
        lambda: "https://api.confident-ai.com",
    )
    api = Api(api_key="  confident_us_proj_test123  ")
    assert api.api_key == "confident_us_proj_test123"
    assert api._headers["CONFIDENT-API-KEY"] == "confident_us_proj_test123"
