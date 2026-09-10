"""`deepeval set-azure-openai, set-azure-openai-embedding, unset-azure-openai, unset-azure-openai-embedding` commands."""

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
    EmbeddingKeyValues,
    ModelKeyValues,
)


@app.command(name="set-azure-openai")
def set_azure_openai_env(
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
            "Prompt for AZURE_OPENAI_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider.",
    ),
    api_version: Optional[str] = typer.Option(
        None,
        "-v",
        "--api-version",
        help="Azure OpenAI API version (passed to the Azure OpenAI client).",
    ),
    model_version: Optional[str] = typer.Option(
        None, "-V", "--model-version", help="Azure model version"
    ),
    deployment_name: Optional[str] = typer.Option(
        None, "-d", "--deployment-name", help="Azure OpenAI deployment name"
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
            typer.prompt("Azure OpenAI API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)
    api_version = coerce_blank_to_none(api_version)
    deployment_name = coerce_blank_to_none(deployment_name)
    model_version = coerce_blank_to_none(model_version)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_AZURE_OPENAI)
        if model is not None:
            settings.AZURE_MODEL_NAME = model
        if api_key is not None:
            settings.AZURE_OPENAI_API_KEY = api_key
        if base_url is not None:
            settings.AZURE_OPENAI_ENDPOINT = base_url
        if api_version is not None:
            settings.OPENAI_API_VERSION = api_version
        if deployment_name is not None:
            settings.AZURE_DEPLOYMENT_NAME = deployment_name
        if model_version is not None:
            settings.AZURE_MODEL_VERSION = model_version

    handled, path, updates = edit_ctx.result

    effective_model = settings.AZURE_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Azure OpenAI model name is not set. Pass --model (or set AZURE_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using Azure OpenAI's `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-azure-openai")
def unset_azure_openai_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Azure OpenAI–related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove AZURE_OPENAI_API_KEY from the dotenv store.",
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
        settings.AZURE_OPENAI_ENDPOINT = None
        settings.OPENAI_API_VERSION = None
        settings.AZURE_DEPLOYMENT_NAME = None
        settings.AZURE_MODEL_NAME = None
        settings.AZURE_MODEL_VERSION = None
        settings.USE_AZURE_OPENAI = None
        if clear_secrets:
            settings.AZURE_OPENAI_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Azure OpenAI environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "Azure OpenAI has been unset. No active provider is configured. Set one with the CLI, or add credentials to .env[.local]."
            )


@app.command(name="set-azure-openai-embedding")
def set_azure_openai_embedding_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider (e.g., `gpt-4.1`).",
    ),
    deployment_name: Optional[str] = typer.Option(
        None,
        "-d",
        "--deployment-name",
        help="Azure embedding deployment name",
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
    deployment_name = coerce_blank_to_none(deployment_name)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(
            EmbeddingKeyValues.USE_AZURE_OPENAI_EMBEDDING
        )
        if model is not None:
            settings.AZURE_EMBEDDING_MODEL_NAME = model
        if deployment_name is not None:
            settings.AZURE_EMBEDDING_DEPLOYMENT_NAME = deployment_name

    handled, path, updates = edit_ctx.result

    effective_model = settings.AZURE_EMBEDDING_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Azure OpenAI embedding model name is not set. Pass --model (or set AZURE_EMBEDDING_MODEL_NAME).",
            param_hint="--model",
        )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using Azure OpenAI embedding model `{escape(effective_model)}` for all evals that require text embeddings."
        ),
    )


@app.command(name="unset-azure-openai-embedding")
def unset_azure_openai_embedding_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Azure OpenAI embedding related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
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
        settings.AZURE_EMBEDDING_MODEL_NAME = None
        settings.AZURE_EMBEDDING_DEPLOYMENT_NAME = None
        settings.USE_AZURE_OPENAI_EMBEDDING = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed Azure OpenAI embedding environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "Azure OpenAI embedding has been unset. No active provider is configured. Set one with the CLI, or add credentials to .env[.local]."
            )
