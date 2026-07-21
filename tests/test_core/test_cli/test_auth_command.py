from typing import Optional

import pytest

from deepeval.cli.auth.api import (
    CliAuthorization,
    CliOnboardingContext,
    CliQuestionnaire,
    DynamicNewUserOnboardingRequest,
    NewUserOnboardingRequest,
)
from deepeval.cli.auth.command import (
    _complete_browser_cli_login,
    _get_pasted_api_key_warnings,
    _prompt_dynamic_questionnaire,
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


def test_dynamic_questionnaire_parses_and_serializes_answers() -> None:
    questionnaire = CliQuestionnaire.model_validate(
        {
            "version": 1,
            "questions": [
                {
                    "id": "organizationName",
                    "type": "text",
                    "prompt": "Organization name",
                    "required": True,
                    "maxLength": 200,
                },
                {
                    "id": "userFacing",
                    "type": "single_select",
                    "prompt": "User facing?",
                    "required": True,
                    "options": [
                        {"label": "Yes", "value": True},
                        {"label": "No", "value": False},
                    ],
                },
            ],
        }
    )
    request = DynamicNewUserOnboardingRequest(
        questionnaire_version=questionnaire.version,
        questionnaire_answers={
            "organizationName": "Acme",
            "userFacing": False,
        },
    )

    assert questionnaire.questions[0].max_length == 200
    assert request.to_payload() == {
        "questionnaireVersion": 1,
        "questionnaireAnswers": {
            "organizationName": "Acme",
            "userFacing": False,
        },
    }


def test_dynamic_questionnaire_renders_exclusive_and_custom_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questionnaire = CliQuestionnaire.model_validate(
        {
            "version": 1,
            "questions": [
                {
                    "id": "organizationName",
                    "type": "text",
                    "prompt": "Organization name",
                    "required": True,
                    "defaultValue": "Acme",
                },
                {
                    "id": "userFacing",
                    "type": "single_select",
                    "prompt": "User facing?",
                    "required": True,
                    "options": [
                        {"label": "Yes", "value": True},
                        {"label": "No", "value": False},
                    ],
                },
                {
                    "id": "externalResources",
                    "type": "multi_select",
                    "prompt": "External resources?",
                    "required": True,
                    "minSelections": 1,
                    "options": [
                        {"label": "RAG", "value": "RAG"},
                        {
                            "label": "None",
                            "value": "NONE",
                            "exclusive": True,
                        },
                    ],
                },
                {
                    "id": "description",
                    "type": "multi_select",
                    "prompt": "Use cases?",
                    "required": True,
                    "options": [
                        {
                            "label": "Something else",
                            "value": "CUSTOM",
                            "acceptsCustomValue": True,
                            "customPrompt": "Describe your use case",
                        }
                    ],
                },
            ],
        }
    )
    checkbox_selections = iter([["NONE", "RAG"], ["RAG"], ["CUSTOM"]])
    monkeypatch.setattr(
        "deepeval.cli.auth.command.prompt_text",
        lambda *_args, **_kwargs: "Acme",
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command.prompt_select",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command.prompt_checkbox",
        lambda *_args, **_kwargs: next(checkbox_selections),
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command._prompt_required",
        lambda *_args, **_kwargs: "Customer support",
    )

    assert _prompt_dynamic_questionnaire(questionnaire) == {
        "organizationName": "Acme",
        "userFacing": False,
        "externalResources": ["RAG"],
        "description": ["Customer support"],
    }


def test_new_client_falls_back_when_server_omits_questionnaire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = CliOnboardingContext.model_validate(
        {
            "state": "new_user",
            "user": {"name": "Ada", "email": "ada@example.com"},
            "projects": [],
        }
    )
    captured_request = None

    monkeypatch.setattr(
        "deepeval.cli.auth.command.browser_pairing_login",
        lambda: CliAuthorization(
            setup_token="setup-token", email="ada@example.com"
        ),
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command.get_cli_onboarding_context",
        lambda _token: context,
    )
    required_answers = iter(["Ada", "Acme", "Support Bot"])
    monkeypatch.setattr(
        "deepeval.cli.auth.command._prompt_required",
        lambda *_args, **_kwargs: next(required_answers),
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command._prompt_project_profile",
        lambda _project_name: {
            "development_stage": "PRODUCTION",
            "interaction_type": "SINGLE_TURN",
            "modalities": ["TEXT"],
            "user_facing": False,
            "external_resources": [],
            "description": "RAG Q&A",
        },
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command.typer.confirm",
        lambda *_args, **_kwargs: True,
    )

    def complete(_token, request, *, idempotency_key):
        nonlocal captured_request
        captured_request = request
        return "api-key"

    monkeypatch.setattr(
        "deepeval.cli.auth.command.complete_cli_onboarding", complete
    )

    assert _complete_browser_cli_login() == "api-key"
    assert isinstance(captured_request, NewUserOnboardingRequest)


def test_new_client_submits_server_questionnaire_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = CliOnboardingContext.model_validate(
        {
            "state": "new_user",
            "user": {"name": "Ada", "email": "ada@example.com"},
            "projects": [],
            "questionnaire": {
                "version": 1,
                "questions": [
                    {
                        "id": "organizationName",
                        "type": "text",
                        "prompt": "Organization name",
                        "required": True,
                    },
                    {
                        "id": "projectName",
                        "type": "text",
                        "prompt": "Project name",
                        "required": True,
                    },
                ],
            },
        }
    )
    answers = {
        "organizationName": "Acme",
        "projectName": "Support Bot",
    }
    captured_request = None

    monkeypatch.setattr(
        "deepeval.cli.auth.command.browser_pairing_login",
        lambda: CliAuthorization(setup_token="setup-token"),
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command.get_cli_onboarding_context",
        lambda _token: context,
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command._prompt_dynamic_questionnaire",
        lambda _questionnaire: answers,
    )
    monkeypatch.setattr(
        "deepeval.cli.auth.command.typer.confirm",
        lambda *_args, **_kwargs: True,
    )

    def complete(_token, request, *, idempotency_key):
        nonlocal captured_request
        captured_request = request
        return "api-key"

    monkeypatch.setattr(
        "deepeval.cli.auth.command.complete_cli_onboarding", complete
    )

    assert _complete_browser_cli_login() == "api-key"
    assert isinstance(captured_request, DynamicNewUserOnboardingRequest)
    assert captured_request.to_payload() == {
        "questionnaireVersion": 1,
        "questionnaireAnswers": answers,
    }
