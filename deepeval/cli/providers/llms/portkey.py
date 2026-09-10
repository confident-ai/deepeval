"""`deepeval set-portkey, unset-portkey` commands."""

import typer
from typing import Optional
from rich import print
from rich.markup import escape

from deepeval.cli.app import app
from deepeval.cli.utils import (
    coerce_blank_to_none,
    handle_save_result as _handle_save_result,
    is_openai_configured,
)
from deepeval.config.settings import get_settings
from deepeval.key_handler import (
    ModelKeyValues,
)


@app.command(name="set-portkey")
def set_portkey_model_env(
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
            "Prompt for PORTKEY_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "-P",
        "--provider",
        help="Override the PORTKEY_PROVIDER_NAME.",
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
            typer.prompt("Portkey API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)
    provider = coerce_blank_to_none(provider)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_PORTKEY_MODEL)
        if model is not None:
            settings.PORTKEY_MODEL_NAME = model
        if api_key is not None:
            settings.PORTKEY_API_KEY = api_key
        if base_url is not None:
            settings.PORTKEY_BASE_URL = base_url
        if provider is not None:
            settings.PORTKEY_PROVIDER_NAME = provider

    handled, path, updates = edit_ctx.result

    effective_model = settings.PORTKEY_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Portkey model name is not set. Pass --model (or set PORTKEY_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using Portkey `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-portkey")
def unset_portkey_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Portkey related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove PORTKEY_API_KEY from the dotenv store.",
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
        settings.USE_PORTKEY_MODEL = None
        settings.PORTKEY_MODEL_NAME = None
        settings.PORTKEY_BASE_URL = None
        settings.PORTKEY_PROVIDER_NAME = None
        if clear_secrets:
            settings.PORTKEY_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Portkey model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The Portkey model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )
