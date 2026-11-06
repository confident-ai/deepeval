"""`deepeval set-openai, unset-openai` commands."""

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


@app.command(name="set-openai")
def set_openai_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider (e.g., `gpt-4.1`).",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for OPENAI_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    cost_per_input_token: Optional[float] = typer.Option(
        None,
        "-i",
        "--cost-per-input-token",
        help=(
            "USD per input token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
        ),
    ),
    cost_per_output_token: Optional[float] = typer.Option(
        None,
        "-o",
        "--cost-per-output-token",
        help=(
            "USD per output token override used for cost tracking. Preconfigured for known models; "
            "REQUIRED if you use a custom/unknown model."
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
    """
    Configure OpenAI as the active LLM provider.

    What this does:
    - Sets the active provider flag to `USE_OPENAI_MODEL`.
    - Persists the selected model name and any cost overrides in the JSON store.
    - secrets are never written to `.deepeval/.deepeval` (JSON).

    Pricing rules:
    - If `model` is a known OpenAI model, you may omit costs (built‑in pricing is used).
    - If `model` is custom/unsupported, you must provide both
      `--cost-per-input-token` and `--cost-per-output-token`.

    Secrets & saving:

    - If you run with --prompt-api-key, DeepEval will set OPENAI_API_KEY for this session.
    - If --save=dotenv[:path] is used (or DEEPEVAL_DEFAULT_SAVE is set), the key will be written to that dotenv file (plaintext).

    Secrets are never written to .deepeval/.deepeval (legacy JSON store).

    Args:
        --model: OpenAI model name, such as `gpt-4o-mini`.
        --prompt-api-key: Prompt interactively for OPENAI_API_KEY (input hidden). Avoids putting secrets on the command line (shell history/process args). Not suitable for CI.
        --cost-per-input-token: USD per input token (optional for known models).
        --cost-per-output-token: USD per output token (optional for known models).
        --save: Persist config (and supported secrets) to a dotenv file; format `dotenv[:path]`.
        --quiet: Suppress printing to the terminal.

    Example:
        deepeval set-openai \\
          --model gpt-4o-mini \\
          --cost-per-input-token 0.0005 \\
          --cost-per-output-token 0.0015 \\
          --save dotenv:.env.local
    """
    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt("OpenAI API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_OPENAI_MODEL)
        if model is not None:
            settings.OPENAI_MODEL_NAME = model
        if api_key is not None:
            settings.OPENAI_API_KEY = api_key
        if cost_per_input_token is not None:
            settings.OPENAI_COST_PER_INPUT_TOKEN = cost_per_input_token
        if cost_per_output_token is not None:
            settings.OPENAI_COST_PER_OUTPUT_TOKEN = cost_per_output_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.OPENAI_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "OpenAI model name is not set. Pass --model (or set OPENAI_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using OpenAI's `{escape(effective_model)}` "
            "for all evals that require an LLM."
        ),
    )


@app.command(name="unset-openai")
def unset_openai_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the OpenAI related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove OPENAI_API_KEY from the dotenv store.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    """
    Unset OpenAI as the active provider.

    Behavior:
    - Removes OpenAI keys (model, costs, toggle) from the JSON store.
    - If `--save` is provided, removes those keys from the specified dotenv file.
    - After unsetting, if `OPENAI_API_KEY` is still set in the environment,
      OpenAI may still be usable by default. Otherwise, no active provider is configured.

    Args:
        --save: Remove OpenAI keys from the given dotenv file as well.
        --clear-secrets: Removes OPENAI_API_KEY from the dotenv store
        --quiet: Suppress printing to the terminal

    Example:
        deepeval unset-openai --save dotenv:.env.local
    """

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        settings.OPENAI_MODEL_NAME = None
        settings.OPENAI_COST_PER_INPUT_TOKEN = None
        settings.OPENAI_COST_PER_OUTPUT_TOKEN = None
        settings.USE_OPENAI_MODEL = None
        if clear_secrets:
            settings.OPENAI_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed OpenAI environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "OpenAI has been unset. No active provider is configured. "
                "Set one with the CLI, or add credentials to .env[.local]."
            )
