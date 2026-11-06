"""`deepeval set-nace, unset-nace` commands."""

import typer
from typing import Optional
from rich import print
from rich.markup import escape

from deepeval.cli.app import app
from deepeval.cli.utils import (
    coerce_blank_to_none,
    handle_save_result as _handle_save_result,
)
from deepeval.config.eval_mode import (
    DECISION_PROVIDER_ENV_VAR,
    EVAL_MODE_ENV_VAR,
    DecisionProvider,
    EvalMode,
    normalize_decision_provider,
)
from deepeval.config.settings import get_settings
from deepeval.models.system_one.constants import (
    DEFAULT_NACE_MODEL,
    NACE_BASE_URL,
)
from deepeval.models.system_one.nace_model import is_openrouter


@app.command(name="set-nace")
def set_nace_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help=f"Nace Drex model (default: {DEFAULT_NACE_MODEL}, which reads "
        f"text and images). A text-only model refuses images.",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for NACE_API_KEY, or OPENROUTER_API_KEY when the base "
            "URL is OpenRouter (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help=f"Override the API base URL (default: {NACE_BASE_URL}; "
        "https://openrouter.ai/api for OpenRouter).",
    ),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "-i",
        "--cost-per-input-token",
        help="USD per input token override used for cost tracking. Preconfigured for Drex models.",
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
    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)

    settings = get_settings()
    openrouter = is_openrouter(
        str(base_url or settings.NACE_BASE_URL or NACE_BASE_URL)
    )

    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt(
                "OpenRouter API key" if openrouter else "Nace API key",
                hide_input=True,
            )
        )

    with settings.edit(save=save) as edit_ctx:
        settings.DEEPEVAL_DECISION_PROVIDER = DecisionProvider.NACE.value
        if api_key is not None and openrouter:
            settings.OPENROUTER_API_KEY = api_key
        elif api_key is not None:
            settings.NACE_API_KEY = api_key
        if model is not None:
            settings.NACE_MODEL_NAME = model
        if base_url is not None:
            settings.NACE_BASE_URL = base_url
        if cost_per_input_token is not None:
            settings.NACE_COST_PER_INPUT_TOKEN = cost_per_input_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.NACE_MODEL_NAME or DEFAULT_NACE_MODEL

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Nace Drex `{escape(effective_model)}` now makes "
            f"System One decisions. Pick how it judges with `deepeval "
            f"set-eval-mode {EvalMode.HYBRID}` or `deepeval set-eval-mode "
            f"{EvalMode.SYSTEM_ONE}`."
        ),
    )


@app.command(name="unset-nace")
def unset_nace_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Nace related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove NACE_API_KEY from the dotenv store.",
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
        settings.NACE_MODEL_NAME = None
        settings.NACE_BASE_URL = None
        settings.NACE_COST_PER_INPUT_TOKEN = None
        if (
            normalize_decision_provider(settings.DEEPEVAL_DECISION_PROVIDER)
            is DecisionProvider.NACE
        ):
            settings.DEEPEVAL_DECISION_PROVIDER = None
        if clear_secrets:
            settings.NACE_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Nace environment variables from {path}.",
        tip_msg=None,
    ):
        print(
            f"The Nace configuration has been removed and "
            f"{DECISION_PROVIDER_ENV_VAR} is back to "
            f"{DecisionProvider.TYPESAFE}. Metrics running with "
            f"{EVAL_MODE_ENV_VAR}={EvalMode.HYBRID} or "
            f"{EVAL_MODE_ENV_VAR}={EvalMode.SYSTEM_ONE} now decide with "
            f"TypeSafe AI Jev (`deepeval set-typesafe`), or switch back with "
            f"`deepeval set-eval-mode {EvalMode.LLM}`."
        )
