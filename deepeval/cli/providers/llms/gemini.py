"""`deepeval set-gemini, unset-gemini` commands."""

import typer
from pathlib import Path
from typing import Optional
from rich import print
from rich.markup import escape

from deepeval.cli.app import app
from deepeval.cli.utils import (
    coerce_blank_to_none,
    handle_save_result as _handle_save_result,
    is_openai_configured,
    load_service_account_key_file,
)
from deepeval.config.settings import get_settings
from deepeval.key_handler import (
    ModelKeyValues,
)


@app.command(name="set-gemini")
def set_gemini_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for GOOGLE_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    project: Optional[str] = typer.Option(
        None,
        "-p",
        "--project",
        help="GCP project ID (used by Vertex AI / Gemini when applicable).",
    ),
    location: Optional[str] = typer.Option(
        None,
        "-l",
        "--location",
        help="GCP location/region for Vertex AI (e.g., `us-central1`).",
    ),
    service_account_file: Optional[Path] = typer.Option(
        None,
        "-S",
        "--service-account-file",
        help=("Path to a Google service account JSON key file."),
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Persist CLI parameters as environment variables in a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("Google API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    project = coerce_blank_to_none(project)
    location = coerce_blank_to_none(location)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_GEMINI_MODEL)

        if model is not None:
            settings.GEMINI_MODEL_NAME = model
        if project is not None:
            settings.GOOGLE_CLOUD_PROJECT = project
        if location is not None:
            settings.GOOGLE_CLOUD_LOCATION = location
        if service_account_file is not None:
            settings.GOOGLE_SERVICE_ACCOUNT_KEY = load_service_account_key_file(
                service_account_file
            )
        if api_key is not None:
            settings.GOOGLE_API_KEY = api_key
            settings.GOOGLE_GENAI_USE_VERTEXAI = False
        elif (
            project is not None
            or location is not None
            or service_account_file is not None
        ):
            settings.GOOGLE_GENAI_USE_VERTEXAI = True

    handled, path, updates = edit_ctx.result

    effective_model = settings.GEMINI_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Gemini model name is not set. Pass --model (or set GEMINI_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using Gemini `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-gemini")
def unset_gemini_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Gemini related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove GOOGLE_API_KEY and GOOGLE_SERVICE_ACCOUNT_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.USE_GEMINI_MODEL = None
        settings.GOOGLE_GENAI_USE_VERTEXAI = None
        settings.GOOGLE_CLOUD_PROJECT = None
        settings.GOOGLE_CLOUD_LOCATION = None
        settings.GEMINI_MODEL_NAME = None
        if clear_secrets:
            settings.GOOGLE_API_KEY = None
            settings.GOOGLE_SERVICE_ACCOUNT_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Gemini model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The Gemini model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )
