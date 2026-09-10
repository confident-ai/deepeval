"""`deepeval set-bedrock, unset-bedrock` commands."""

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


@app.command(name="set-bedrock")
def set_bedrock_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    prompt_credentials: bool = typer.Option(
        False,
        "-a",
        "--prompt-credentials",
        help=(
            "Prompt for AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (secret access key input is hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, credentials are written to dotenv in plaintext."
        ),
    ),
    region: Optional[str] = typer.Option(
        None,
        "-r",
        "--region",
        help="AWS region for bedrock (e.g., `us-east-1`).",
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
    access_key_id = None
    secret_access_key = None
    if prompt_credentials:
        access_key_id = coerce_blank_to_none(typer.prompt("AWS Access key Id"))
        secret_access_key = coerce_blank_to_none(
            typer.prompt("AWS Secret Access key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    region = coerce_blank_to_none(region)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_AWS_BEDROCK_MODEL)
        if access_key_id is not None:
            settings.AWS_ACCESS_KEY_ID = access_key_id
        if secret_access_key is not None:
            settings.AWS_SECRET_ACCESS_KEY = secret_access_key
        if model is not None:
            settings.AWS_BEDROCK_MODEL_NAME = model
        if region is not None:
            settings.AWS_BEDROCK_REGION = region
        if cost_per_input_token is not None:
            settings.AWS_BEDROCK_COST_PER_INPUT_TOKEN = cost_per_input_token
        if cost_per_output_token is not None:
            settings.AWS_BEDROCK_COST_PER_OUTPUT_TOKEN = cost_per_output_token

    handled, path, updates = edit_ctx.result

    effective_model = settings.AWS_BEDROCK_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "AWS Bedrock model name is not set. Pass --model (or set AWS_BEDROCK_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using AWS Bedrock `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-bedrock")
def unset_bedrock_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the AWS Bedrock model related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY  from the dotenv store.",
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
        settings.USE_AWS_BEDROCK_MODEL = None
        settings.AWS_BEDROCK_MODEL_NAME = None
        settings.AWS_BEDROCK_REGION = None
        settings.AWS_BEDROCK_COST_PER_INPUT_TOKEN = None
        settings.AWS_BEDROCK_COST_PER_OUTPUT_TOKEN = None
        if clear_secrets:
            settings.AWS_ACCESS_KEY_ID = None
            settings.AWS_SECRET_ACCESS_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed AWS Bedrock model environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The AWS Bedrock model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )
