"""`deepeval set-ollama, set-ollama-embeddings, unset-ollama, unset-ollama-embeddings` commands."""

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


@app.command(name="set-ollama")
def set_ollama_model_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider",
    ),
    base_url: str = typer.Option(
        "http://localhost:11434",
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
    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(ModelKeyValues.USE_LOCAL_MODEL)
        settings.LOCAL_MODEL_API_KEY = "ollama"
        if model is not None:
            settings.OLLAMA_MODEL_NAME = model
        if base_url is not None:
            settings.LOCAL_MODEL_BASE_URL = base_url

    handled, path, updates = edit_ctx.result

    effective_model = settings.OLLAMA_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Ollama model name is not set. Pass --model (or set OLLAMA_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using a local Ollama model `{escape(effective_model)}` for all evals that require an LLM."
        ),
    )


@app.command(name="unset-ollama")
def unset_ollama_model_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Ollama related environment variables from a dotenv file. "
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
        if clear_secrets:
            settings.LOCAL_MODEL_API_KEY = None
        settings.OLLAMA_MODEL_NAME = None
        settings.LOCAL_MODEL_BASE_URL = None
        settings.USE_LOCAL_MODEL = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed local Ollama environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: OpenAI will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The local Ollama model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )


@app.command(name="set-ollama-embeddings")
def set_ollama_embeddings_env(
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier to use for this provider.",
    ),
    base_url: str = typer.Option(
        "http://localhost:11434",
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
    model = coerce_blank_to_none(model)
    base_url = coerce_blank_to_none(base_url)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(EmbeddingKeyValues.USE_LOCAL_EMBEDDINGS)
        settings.LOCAL_EMBEDDING_API_KEY = "ollama"
        if model is not None:
            settings.LOCAL_EMBEDDING_MODEL_NAME = model
        if base_url is not None:
            settings.LOCAL_EMBEDDING_BASE_URL = base_url

    handled, path, updates = edit_ctx.result

    effective_model = settings.LOCAL_EMBEDDING_MODEL_NAME
    if not effective_model:
        raise typer.BadParameter(
            "Ollama embedding model name is not set. Pass --model (or set LOCAL_EMBEDDING_MODEL_NAME).",
            param_hint="--model",
        )
    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using the Ollama embedding model `{escape(effective_model)}` for all evals that require text embeddings."
        ),
    )


@app.command(name="unset-ollama-embeddings")
def unset_ollama_embeddings_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the Ollama embedding related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove LOCAL_EMBEDDING_API_KEY from the dotenv store.",
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
        if clear_secrets:
            settings.LOCAL_EMBEDDING_API_KEY = None
        settings.LOCAL_EMBEDDING_MODEL_NAME = None
        settings.LOCAL_EMBEDDING_BASE_URL = None
        settings.USE_LOCAL_EMBEDDINGS = None

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg="Removed local Ollama embedding environment variables from {path}.",
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                ":raised_hands: Regular OpenAI embeddings will still be used by default because OPENAI_API_KEY is set."
            )
        else:
            print(
                "The local Ollama embedding model configuration has been removed. No model is currently configured, but you can set one with the CLI or add credentials to .env[.local]."
            )
