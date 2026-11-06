"""`deepeval set-local-embeddings, unset-local-embeddings` commands."""

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
)


@app.command(name="set-local-embeddings")
def set_local_embeddings_env(
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
            "Prompt for LOCAL_EMBEDDING_API_KEY (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
        ),
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="Override the API endpoint/base URL used by this provider.",
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
            typer.prompt("Local Embedding Model API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)
        if model is not None:
            settings.LOCAL_EMBEDDING_MODEL_NAME = model
        if base_url is not None:
            settings.LOCAL_EMBEDDING_BASE_URL = base_url
        if api_key is not None:
            settings.LOCAL_EMBEDDING_API_KEY = api_key

    handled, path, updates = edit_ctx.result

    effective_model = settings.LOCAL_EMBEDDING_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Local embedding model name is not set. Pass --model (or set LOCAL_EMBEDDING_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using the local embedding model `{escape(effective_model)}` for all evals that require text embeddings."
        ),
    )


@app.command(name="unset-local-embeddings")
def unset_local_embeddings_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the local embedding related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove LOCAL_MODEL_API_KEY from the dotenv store.",
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
        settings.LOCAL_EMBEDDING_MODEL_NAME = None
        settings.LOCAL_EMBEDDING_BASE_URL = None
        settings.USE_LOCAL_EMBEDDINGS = None
        if clear_secrets:
            settings.LOCAL_EMBEDDING_API_KEY = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed local embedding environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The local embeddings model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )
