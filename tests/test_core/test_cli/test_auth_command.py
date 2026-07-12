from typing import Optional

import pytest

from deepeval.cli.auth.command import _get_pasted_api_key_warnings


@pytest.mark.parametrize(
    ("api_key", "configured_region"),
    [
        ("custom-key-containing-org", "US"),
        ("confident_eu_unknown_secret", "US"),
        ("confident_us_proj_c2VjcmV0", "US"),
        ("confident_us_global_c2VjcmV0", "US"),
        ("confident_eu_proj_c2VjcmV0", None),
    ],
)
def test_pasted_api_key_does_not_warn_without_known_issue(
    api_key: str, configured_region: Optional[str]
) -> None:
    assert _get_pasted_api_key_warnings(api_key, configured_region) == []


def test_pasted_api_key_warns_for_organization_scope() -> None:
    warnings = _get_pasted_api_key_warnings("confident_us_org_c2VjcmV0", "US")

    assert len(warnings) == 1
    assert "organization API key" in warnings[0]


def test_pasted_api_key_warns_for_region_mismatch() -> None:
    warnings = _get_pasted_api_key_warnings("confident_eu_proj_c2VjcmV0", "US")

    assert len(warnings) == 1
    assert "EU region" in warnings[0]
    assert "configured to use US" in warnings[0]


def test_pasted_api_key_warns_for_scope_and_region_mismatch() -> None:
    warnings = _get_pasted_api_key_warnings("confident_eu_org_c2VjcmV0=", "US")

    assert len(warnings) == 2
    assert "EU region" in warnings[0]
    assert "organization API key" in warnings[1]
