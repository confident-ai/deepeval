from typing import Optional

import pytest

from deepeval.cli.auth.api import NewUserOnboardingRequest
from deepeval.cli.auth.command import (
    _get_pasted_api_key_warnings,
    _prompt_project_profile,
)


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


def test_prompt_project_profile_collects_ui_onboarding_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selections = iter(["DEVELOPMENT", "MULTI_TURN", True])
    checkbox_selections = iter(
        [
            ["TEXT", "IMAGE"],
            ["TOOL_CALL", "MCP"],
            ["Chatbot assistant", "CUSTOM"],
        ]
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command.prompt_select",
        lambda *_args, **_kwargs: next(selections),
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command.prompt_checkbox",
        lambda *_args, **_kwargs: next(checkbox_selections),
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command._prompt_required",
        lambda *_args, **_kwargs: "Customer support",
    )

    assert _prompt_project_profile("Support Bot") == {
        "development_stage": "DEVELOPMENT",
        "interaction_type": "MULTI_TURN",
        "modalities": ["TEXT", "IMAGE"],
        "user_facing": True,
        "external_resources": ["TOOL_CALL", "MCP"],
        "description": "Chatbot assistant, Customer support",
    }


def test_new_user_onboarding_serializes_project_profile() -> None:
    request = NewUserOnboardingRequest(
        user_name="Ada",
        organization_name="Acme",
        project_name="Support Bot",
        development_stage="PRODUCTION",
        interaction_type="SINGLE_TURN",
        modalities=["TEXT"],
        user_facing=False,
        external_resources=[],
        description="RAG Q&A",
    )

    assert request.to_payload() == {
        "userName": "Ada",
        "organizationName": "Acme",
        "projectName": "Support Bot",
        "developmentStage": "PRODUCTION",
        "interactionType": "SINGLE_TURN",
        "modalities": ["TEXT"],
        "userFacing": False,
        "externalResources": [],
        "description": "RAG Q&A",
    }
