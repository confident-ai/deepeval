"""`deepeval set-openrouter, unset-openrouter` commands."""

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


@app.command(name="set-openrouter")
def set_openrouter_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider (e.g., `openai/gpt-4.1`).",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for OPENROUTER_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider (default: https://openrouter.ai/api/v1).",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "-t",
        "--temperature",
        help="Override the global TEMPERATURE used by LLM providers (e.g., 0.0 for deterministic behavior).",
    ),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "-i",
        "--cost-per-input-token",
        help=(
            "USD per input token used for cost tracking. "
            "If unset and OpenRouter does not return pricing metadata, "
            "costs will not be calculated."
        ),
    ),
    cost_per_output_token: Optional[float] = typer.Option(
        None,
        "-o",
        "--cost-per-output-token",
        help=(
            "USD per output token used for cost tracking. "
            "If unset and OpenRouter does not return pricing metadata, "
            "costs will not be calculated."
        ),
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
            typer.prompt("OpenRouter API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_OPENROUTER_MODEL)
        if model is not None:
            settings.OPENROUTER_MODEL_NAME = model
        if api_key is not None:
            settings.OPENROUTER_API_KEY = api_key
        if base_url is not None:
            settings.OPENROUTER_BASE_URL = base_url
        if temperature is not None:
            settings.TEMPERATURE = temperature
        if cost_per_input_token is not None:
            settings.OPENROUTER_COST_PER_INPUT_TOKEN = cost_per_input_token
        if cost_per_output_token is not None:
            settings.OPENROUTER_COST_PER_OUTPUT_TOKEN = cost_per_output_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.OPENROUTER_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "OpenRouter model name is not set. Pass --model (or set OPENROUTER_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using OpenRouter `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-openrouter")
def unset_openrouter_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the OpenRouter model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove OPENROUTER_API_KEY from the dotenv store.",
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
        settings.USE_OPENROUTER_MODEL = None
        settings.OPENROUTER_MODEL_NAME = None
        settings.OPENROUTER_BASE_URL = None
        settings.OPENROUTER_COST_PER_INPUT_TOKEN = None
        settings.OPENROUTER_COST_PER_OUTPUT_TOKEN = None
        # Intentionally do NOT touch TEMPERATURE here; it's a global dial.
        if clear_secrets:
            settings.OPENROUTER_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed OpenRouter model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The OpenRouter model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )
