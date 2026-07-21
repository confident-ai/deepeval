"""`deepeval login` / `deepeval logout` Typer commands."""

import re
import webbrowser
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import typer
from rich import print

from deepeval.cli.auth.api import (
    CliMultiSelectQuestion,
    CliQuestionnaire,
    CliTextQuestion,
    DynamicNewUserOnboardingRequest,
    ExistingProjectKeyRequest,
    NewUserOnboardingRequest,
    QuestionnaireAnswer,
)
from deepeval.cli.auth.flow import (
    AuthFlowError,
    browser_pairing_login,
    complete_cli_onboarding,
    get_cli_onboarding_context,
    prompt_checkbox,
    prompt_select,
    prompt_text,
)
from deepeval.cli.diagnose import resolve_setting_source
from deepeval.cli.dotenv_handler import DotenvHandler
from deepeval.cli.utils import (
    PROD,
    WWW,
    coerce_blank_to_none,
    handle_save_result,
    render_confident_banner,
    with_utm,
)
from deepeval.config.settings import dotenv_search_paths, get_settings
from deepeval.config.utils import read_dotenv_file
from deepeval.telemetry import capture_login_event
from deepeval.test_run.test_run import LATEST_TEST_RUN_FILE_PATH
from deepeval.utils import delete_file_if_exists

LOGIN_HELP = (
    "Log in to Confident AI. Opens the platform for authentication, then "
    "completes organization and project setup in the terminal. "
    f"You can still manage project API keys at {with_utm(PROD, medium='cli', content='login_help_text')}. "
    "The key is saved to your environment variables, typically .env.local, "
    "unless a different path is provided with --save."
)

# Keep the manual browser-and-paste flow available as an emergency rollback.
USE_BROWSER_PAIRING_LOGIN = True

REGION_CHOICES = [
    ("🇺🇸 United States (US)", "US"),
    ("🇪🇺 European Union (EU)", "EU"),
]

DEVELOPMENT_STAGE_CHOICES = [
    ("I'm just exploring an idea", "IDEATION"),
    ("My AI app's in development", "DEVELOPMENT"),
    ("My AI app's already in production", "PRODUCTION"),
]
INTERACTION_TYPE_CHOICES = [
    (
        "Single-Turn — Each request is a standalone interaction",
        "SINGLE_TURN",
    ),
    (
        "Multi-Turn — Maintains context throughout a conversation",
        "MULTI_TURN",
    ),
]
MODALITY_CHOICES = [("Text", "TEXT"), ("Image", "IMAGE"), ("Audio", "AUDIO")]
EXTERNAL_RESOURCE_CHOICES = [
    ("Tool calls", "TOOL_CALL"),
    ("MCP", "MCP"),
    ("RAG", "RAG"),
    ("None of these", "NONE"),
]
USE_CASE_CHOICES = [
    (
        "Document extraction / summarization",
        "Document extraction / summarization",
    ),
    ("Chatbot assistant", "Chatbot assistant"),
    ("Coding agent", "Coding agent"),
    ("RAG Q&A", "RAG Q&A"),
    ("Something else", "CUSTOM"),
]

API_KEY_PATTERN = re.compile(
    r"^confident_(?P<region>us|eu)_(?P<scope>org|proj|global)_"
    r"[A-Za-z0-9+/]+={0,2}$",
    re.IGNORECASE,
)


def _prompt_and_persist_region(save: Optional[str]) -> None:
    """Pick the data region before the pairing session is created, so the
    session, the browser page, and all subsequent polling hit the same
    regional backend. Persisted like `deepeval set-confident-region`."""
    settings = get_settings()
    current = (settings.CONFIDENT_REGION or "US").upper()
    choices = sorted(REGION_CHOICES, key=lambda choice: choice[1] != current)
    region = prompt_select("Select your Confident AI data region:", choices)
    if region == settings.CONFIDENT_REGION:
        return
    with settings.edit(save=save):
        settings.CONFIDENT_REGION = region


def _print_api_key_location() -> None:
    settings_url = PROD
    print(
        "Find your project API key at "
        f"[link={settings_url}]{settings_url}[/link] under "
        "[bold]Project Settings > API Keys[/bold]."
    )


def _get_pasted_api_key_warnings(
    api_key: str, configured_region: Optional[str]
) -> List[str]:
    match = API_KEY_PATTERN.fullmatch(api_key.strip())
    if match is None:
        return []

    key_region = match.group("region").upper()
    key_scope = match.group("scope").lower()
    warnings: List[str] = []

    if configured_region and key_region != configured_region.upper():
        warnings.append(
            f"This API key is for the {key_region} region, but DeepEval is "
            f"configured to use {configured_region.upper()}. The region "
            "configured in DeepEval must match the key's region in "
            "Confident AI."
        )

    if key_scope == "org":
        warnings.append(
            "This is an organization API key, which cannot be used to log in "
            "to DeepEval. Use a project API key from Project Settings > API "
            "Keys instead."
        )

    return warnings


def _warn_for_pasted_api_key(api_key: str) -> None:
    configured_region = get_settings().CONFIDENT_REGION
    for warning in _get_pasted_api_key_warnings(api_key, configured_region):
        print(f"⚠️  {warning}")


def _prompt_paste_api_key() -> str:
    while True:
        api_key = coerce_blank_to_none(
            typer.prompt("🔐 Enter your project API key", hide_input=True)
        )
        if api_key:
            _warn_for_pasted_api_key(api_key)
            return api_key
        print("❌ Project API key cannot be empty. Please try again.\n")


def _open_platform_and_prompt_api_key() -> str:
    platform_url = with_utm(
        PROD, medium="cli", content="login_api_key_browser_open"
    )
    print("\n🌐 Opening Confident AI in your browser...")
    webbrowser.open(platform_url)
    print(
        "(open this link if your browser did not open: "
        f"[link={platform_url}]{platform_url}[/link])"
    )
    return _prompt_paste_api_key()


def _prompt_required(message: str, default: Optional[str] = None) -> str:
    while True:
        value = coerce_blank_to_none(prompt_text(message, default=default))
        if value:
            return value
        print(f"❌ {message} cannot be empty. Please try again.\n")


def _prompt_questionnaire_text(question: CliTextQuestion) -> str:
    while True:
        value = prompt_text(question.prompt, default=question.default_value)
        value = value.strip()
        if question.required and not value:
            print(f"❌ {question.prompt} cannot be empty. Please try again.\n")
            continue
        if question.max_length is not None and len(value) > question.max_length:
            print(
                f"❌ {question.prompt} must be at most "
                f"{question.max_length} characters."
            )
            continue
        return value


def _prompt_dynamic_questionnaire(
    questionnaire: CliQuestionnaire,
) -> Dict[str, QuestionnaireAnswer]:
    answers: Dict[str, QuestionnaireAnswer] = {}

    for question in questionnaire.questions:
        if isinstance(question, CliTextQuestion):
            answers[question.id] = _prompt_questionnaire_text(question)
            continue

        choices = [(option.label, option.value) for option in question.options]
        if question.type == "single_select":
            if question.default_value is not None:
                choices.sort(
                    key=lambda choice: choice[1] != question.default_value
                )
            answers[question.id] = prompt_select(question.prompt, choices)
            continue

        if not isinstance(question, CliMultiSelectQuestion):
            raise AuthFlowError(
                f"Unsupported questionnaire question type: {question.type}."
            )

        while True:
            minimum = question.min_selections or (1 if question.required else 0)
            selected = prompt_checkbox(
                question.prompt,
                choices,
                min_selections=minimum,
            )
            if len(selected) < minimum:
                print(f"❌ Select at least {minimum} option(s).")
                continue

            exclusive_values = {
                option.value for option in question.options if option.exclusive
            }
            if len(selected) > 1 and any(
                value in exclusive_values for value in selected
            ):
                print(
                    "❌ An exclusive option cannot be combined with "
                    "another selection."
                )
                continue
            break

        for option in question.options:
            if option.accepts_custom_value and option.value in selected:
                custom_value = _prompt_required(
                    option.custom_prompt or "Please specify"
                )
                selected = [
                    custom_value if value == option.value else value
                    for value in selected
                ]
        answers[question.id] = selected

    return answers


def _prompt_project_profile(project_name: str) -> Dict[str, Any]:
    print(f"\n[bold]Tell us more about {project_name}.[/bold]")
    development_stage = prompt_select(
        "What stage is your AI application in?",
        DEVELOPMENT_STAGE_CHOICES,
    )
    interaction_type = prompt_select(
        "How does your application interact with users?",
        INTERACTION_TYPE_CHOICES,
    )
    modalities = prompt_checkbox(
        "Which modalities does your application support?",
        MODALITY_CHOICES,
    )
    user_facing = prompt_select(
        "Does your application interact directly with end users?",
        [("Yes", True), ("No", False)],
    )

    while True:
        external_resources = prompt_checkbox(
            "How does your application use external data or tools?",
            EXTERNAL_RESOURCE_CHOICES,
        )
        if "NONE" not in external_resources or len(external_resources) == 1:
            break
        print("❌ 'None of these' cannot be combined with another option.")
    if external_resources == ["NONE"]:
        external_resources = []

    use_cases = prompt_checkbox(
        "Which use cases best describe your application?",
        USE_CASE_CHOICES,
    )
    if "CUSTOM" in use_cases:
        custom_use_case = _prompt_required("Describe your use case")
        use_cases = [
            custom_use_case if use_case == "CUSTOM" else use_case
            for use_case in use_cases
        ]

    return {
        "development_stage": development_stage,
        "interaction_type": interaction_type,
        "modalities": modalities,
        "user_facing": user_facing,
        "external_resources": external_resources,
        "description": ", ".join(use_cases),
    }


def _complete_browser_cli_login() -> Optional[str]:
    authorization = browser_pairing_login()
    if authorization is None:
        return None

    try:
        context = get_cli_onboarding_context(authorization.setup_token)
        request: Union[
            NewUserOnboardingRequest,
            DynamicNewUserOnboardingRequest,
            ExistingProjectKeyRequest,
        ]

        if context.state == "new_user":
            print("\n[bold]Let's set up your workspace.[/bold]")
            if context.questionnaire is not None:
                questionnaire_answers = _prompt_dynamic_questionnaire(
                    context.questionnaire
                )
                organization_name = questionnaire_answers.get(
                    "organizationName"
                )
                project_name = questionnaire_answers.get("projectName")
                if not isinstance(organization_name, str) or not isinstance(
                    project_name, str
                ):
                    raise AuthFlowError(
                        "The server questionnaire did not collect the "
                        "required organization and project names."
                    )
                request = DynamicNewUserOnboardingRequest(
                    questionnaire_version=context.questionnaire.version,
                    questionnaire_answers=questionnaire_answers,
                )
            else:
                existing_name = context.user.name if context.user else None
                name_default = (
                    existing_name
                    if existing_name and existing_name != "New User"
                    else None
                )
                user_name = _prompt_required("Your name", default=name_default)
                organization_name = _prompt_required("Organization name")
                project_name = _prompt_required(
                    "Project name", default="My First Project"
                )
                project_profile = _prompt_project_profile(project_name)
                request = NewUserOnboardingRequest(
                    user_name=user_name,
                    organization_name=organization_name,
                    project_name=project_name,
                    **project_profile,
                )
            print(
                "\nYour organization and project will be created as "
                f"[bold]{organization_name}[/bold] / "
                f"[bold]{project_name}[/bold]."
            )
            if not typer.confirm("Continue?", default=True):
                raise AuthFlowError("Setup cancelled.")
        else:
            projects = [
                project
                for project in context.projects
                if project.can_create_api_key
            ]
            if not projects:
                raise AuthFlowError(
                    "You do not have permission to create an API key for any "
                    "project in this organization."
                )
            project_id = prompt_select(
                "Select a project:",
                [(project.name, project.id) for project in projects],
            )
            request = ExistingProjectKeyRequest(project_id=project_id)

        return complete_cli_onboarding(
            authorization.setup_token,
            request,
            idempotency_key=str(uuid4()),
        )
    except AuthFlowError as error:
        print(f"\n⚠️  CLI onboarding could not be completed: {error}")
        return None


def _resolve_login_key(save: Optional[str]) -> str:
    if not USE_BROWSER_PAIRING_LOGIN:
        return _open_platform_and_prompt_api_key()

    method = prompt_select(
        "How would you like to log in to Confident AI?",
        [
            ("Log in via your browser", "browser"),
            ("Paste a project API key", "paste"),
        ],
    )

    if method == "paste":
        _print_api_key_location()
        return _prompt_paste_api_key()

    _prompt_and_persist_region(save)
    key = _complete_browser_cli_login()
    if key:
        return key
    print("\nNo problem — paste a project API key from the platform instead.")
    _print_api_key_location()
    return _prompt_paste_api_key()


def login_command(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Where to persist settings. Format: dotenv[:path]. Defaults to .env.local. If omitted, login still writes to .env.local.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Log in non-interactively with a project API key instead of the prompts.",
    ),
):
    api_key = coerce_blank_to_none(api_key)

    with capture_login_event() as span:
        completed = False
        try:
            settings = get_settings()
            save = save or settings.DEEPEVAL_DEFAULT_SAVE or "dotenv:.env.local"

            # Resolve the key from the CLI flag or the active interactive flow.
            if api_key is not None:
                key = api_key
                _warn_for_pasted_api_key(key)
            else:
                key = _resolve_login_key(save)

            with settings.edit(save=save) as edit_ctx:
                settings.CONFIDENT_API_KEY = key

            handled, path, updated = edit_ctx.result

            if updated:
                if not handled and save is not None:
                    # invalid --save format (unsupported)
                    print(
                        "Unsupported --save option. Use --save=dotenv[:path]."
                    )
                elif path:
                    # persisted to a file
                    print(
                        f"Saved environment variables to {path} (ensure it's git-ignored)."
                    )

            completed = True
            print("\n[bold]Welcome to[/bold]")
            render_confident_banner()
            print(
                "🎉🥳 Congratulations! You've successfully logged in! :raising_hands:"
            )
            quickstart_url = with_utm(
                f"{WWW}/docs/llm-evaluation/quickstart",
                medium="cli",
                content="login_success_quickstart",
            )
            print(
                "You're now using DeepEval with [rgb(106,0,255)]Confident AI[/rgb(106,0,255)]. "
                "Follow our quickstart tutorial here: "
                f"[bold][link={quickstart_url}]{quickstart_url}[/link][/bold]"
            )
        except Exception as e:
            completed = False
            print(f"Login failed: {e}")
        finally:
            if getattr(span, "set_attribute", None):
                span.set_attribute("completed", completed)


def logout_command(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Where to remove the saved key from. Use format dotenv[:path]. If omitted, uses DEEPEVAL_DEFAULT_SAVE or .env.local. The JSON keystore is always cleared.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    """
    Log out of Confident AI.

    Behavior:
    - Always clears the Confident API key from the JSON keystore and process env.
    - Removes the key from every dotenv file deepeval auto-loads (.env,
      .env.{APP_ENV}, .env.local), plus the --save target if one is given.
    - If the key is exported by the shell itself, deepeval cannot unset it;
      a warning with the fix is printed instead of a success message.
    """
    settings = get_settings()

    # Capture provenance before clearing anything: once files are wiped we
    # can no longer tell a shell export apart from a file-loaded value.
    key_source = resolve_setting_source("CONFIDENT_API_KEY")

    save = save or settings.DEEPEVAL_DEFAULT_SAVE or "dotenv:.env.local"
    with settings.edit(save=save) as edit_ctx:
        settings.CONFIDENT_API_KEY = None

    handled, path, updated = edit_ctx.result

    # The --save target is a single file; also sweep the rest of the dotenv
    # search path so a lower-precedence file (e.g. .env) can't silently log
    # the user back in on the next run.
    for dotenv_path in dotenv_search_paths():
        if dotenv_path.is_file() and "CONFIDENT_API_KEY" in read_dotenv_file(
            dotenv_path
        ):
            DotenvHandler(dotenv_path).unset(["CONFIDENT_API_KEY"])
            if not quiet:
                print(f"Removed Confident AI key(s) from {dotenv_path}.")

    shell_export = key_source == "process environment"

    if (
        handle_save_result(
            handled=handled,
            path=path,
            updates=updated,
            save=save,
            quiet=quiet,
            updated_msg="Removed Confident AI key(s) from {path}.",
            tip_msg=None,
        )
        and not shell_export
    ):
        print("\n🎉🥳 You've successfully logged out! :raising_hands: ")

    if shell_export and not quiet:
        print(
            "\n[yellow]⚠  CONFIDENT_API_KEY is exported by your shell, which "
            "deepeval cannot unset — this terminal will still be logged "
            "in.[/yellow]"
        )
        print(
            "   Finish logging out with: "
            "[bold cyan]unset CONFIDENT_API_KEY[/bold cyan] "
            "[dim](and remove it from your shell profile if it's set "
            "there)[/dim]"
        )

    delete_file_if_exists(LATEST_TEST_RUN_FILE_PATH)
