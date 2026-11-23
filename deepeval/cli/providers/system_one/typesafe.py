"""`deepeval set-typesafe, unset-typesafe` commands."""

import typer
from typing import Optional
from rich import print
from rich.markup import escape

from deepeval.cli.app import app
from deepeval.cli.utils import (
    coerce_blank_to_none,
    handle_save_result as _handle_save_result,
)
from deepeval.config.eval_mode import EVAL_MODE_ENV_VAR, EvalMode
from deepeval.config.settings import get_settings
from deepeval.models.system_one.constants import DEFAULT_TYPESAFE_MODEL


@app.command(name="set-typesafe")
def set_typesafe_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help=f"TypeSafe AI System One model (default: {DEFAULT_TYPESAFE_MODEL})",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for TYPESAFE_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "-i",
        "--cost-per-input-token",
        help="USD per input token override used for cost tracking. Preconfigured for Jev models.",
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
            typer.prompt("TypeSafe AI API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        if api_key is not None:
            settings.TYPESAFE_API_KEY = api_key
        if model is not None:
            settings.TYPESAFE_MODEL_NAME = model
        if cost_per_input_token is not None:
            settings.TYPESAFE_COST_PER_INPUT_TOKEN = cost_per_input_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.TYPESAFE_MODEL_NAME or DEFAULT_TYPESAFE_MODEL

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: TypeSafe AI `{escape(effective_model)}` is configured. "
            f"Pick how it judges with `deepeval set-eval-mode "
            f"{EvalMode.HYBRID}` or `deepeval set-eval-mode "
            f"{EvalMode.SYSTEM_ONE}`."
        ),
    )


@app.command(name="unset-typesafe")
def unset_typesafe_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the TypeSafe AI related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove TYPESAFE_API_KEY from the dotenv store.",
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
        settings.TYPESAFE_MODEL_NAME = None
        settings.TYPESAFE_COST_PER_INPUT_TOKEN = None
        if clear_secrets:
            settings.TYPESAFE_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed TypeSafe AI environment variables from {path}.",
        tip_msg=None,
    ):
        print(
            f"The TypeSafe AI configuration has been removed. Metrics running "
            f"with {EVAL_MODE_ENV_VAR}={EvalMode.HYBRID} or "
            f"{EVAL_MODE_ENV_VAR}={EvalMode.SYSTEM_ONE} will now fail until "
            f"TYPESAFE_API_KEY is set again or you switch back with "
            f"`deepeval set-eval-mode {EvalMode.LLM}`."
        )
