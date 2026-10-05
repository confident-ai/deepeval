"""`deepeval set-stt / unset-stt` commands."""

from enum import Enum
from typing import Optional

import typer

from deepeval.cli.app import app
from deepeval.cli.utils import USE_STT_KEYS
from deepeval.cli.providers._speech import (
    _set_speech_provider,
    _unset_speech_provider,
)


class STTProviders(str, Enum):
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    CARTESIA = "cartesia"
    DEEPGRAM = "deepgram"
    ASSEMBLYAI = "assemblyai"


@app.command(name="set-stt")
def set_stt_env(
    provider: STTProviders = typer.Argument(
        ...,
        help="Speech-to-text provider to use in voice simulations.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model identifier for this provider (e.g., `nova-3`). Defaults to the provider's own default.",
    ),
    prompt_api_key: bool = typer.Option(
        False,
        "-k",
        "--prompt-api-key",
        help=(
            "Prompt for the provider's API key (input hidden). Not suitable for CI. "
            "If --save (or DEEPEVAL_DEFAULT_SAVE) is used, the key is written to dotenv in plaintext."
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
    Configure the active speech-to-text provider.

    What this does:
    - Sets the active provider flag (e.g. `USE_DEEPGRAM_STT`), clearing the other STT flags.
    - Persists `DEEPEVAL_STT_MODEL` in the JSON store when `--model` is given.
    - Leaves the LLM, embedding and TTS selections alone.

    Example:
        deepeval set-stt assemblyai --model universal-3-5-pro --save dotenv:.env.local
    """
    _set_speech_provider(
        family="STT",
        provider=provider.value,
        model=model,
        prompt_api_key=prompt_api_key,
        save=save,
        quiet=quiet,
    )


@app.command(name="unset-stt")
def unset_stt_env(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Remove only the STT related environment variables from a dotenv file. "
        "Usage: --save=dotenv[:path] (default: .env.local)",
    ),
    clear_secrets: bool = typer.Option(
        False,
        "-x",
        "--clear-secrets",
        help="Also remove the speech provider API keys from the dotenv store (OPENAI_API_KEY is left alone).",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    """
    Unset the active speech-to-text provider.

    Voice simulations fall back to OpenAI, exactly as they do when no STT
    provider was ever configured.

    Example:
        deepeval unset-stt --save dotenv:.env.local
    """
    _unset_speech_provider(
        family="STT",
        flags=USE_STT_KEYS,
        clear_secrets=clear_secrets,
        save=save,
        quiet=quiet,
    )
