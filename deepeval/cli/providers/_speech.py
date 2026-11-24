"""Helpers shared by the `set-tts` / `set-stt` command families."""

from typing import List, Optional

import typer
from rich import print
from rich.markup import escape

from deepeval.cli.utils import (
    coerce_blank_to_none,
    handle_save_result as _handle_save_result,
    is_openai_configured,
)
from deepeval.config.settings import get_settings
from deepeval.key_handler import SpeechKeyValues


# provider -> (display name, API key setting)
_SPEECH_PROVIDER_INFO = {
    "openai": ("OpenAI", "OPENAI_API_KEY"),
    "elevenlabs": ("ElevenLabs", "ELEVENLABS_API_KEY"),
    "cartesia": ("Cartesia", "CARTESIA_API_KEY"),
    "deepgram": ("Deepgram", "DEEPGRAM_API_KEY"),
    "assemblyai": ("AssemblyAI", "ASSEMBLYAI_API_KEY"),
}


def _set_speech_provider(
    *,
    family: str,
    provider: str,
    model: Optional[str],
    prompt_api_key: bool,
    save: Optional[str],
    quiet: bool,
) -> None:
    label, key_field = _SPEECH_PROVIDER_INFO[provider]
    flag = SpeechKeyValues[f"USE_{provider.upper()}_{family}"]
    model_field = f"DEEPEVAL_{family}_MODEL"

    api_key = None
    if prompt_api_key:
        api_key = coerce_blank_to_none(
            typer.prompt(f"{label} API key", hide_input=True)
        )

    model = coerce_blank_to_none(model)

    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        edit_ctx.switch_model_provider(flag)
        if model is not None:
            setattr(settings, model_field, model)
        if api_key is not None:
            setattr(settings, key_field, api_key)

    handled, path, updates = edit_ctx.result

    effective_model = getattr(settings, model_field)
    model_phrase = (
        f"`{escape(effective_model)}`"
        if effective_model
        else "with its default model"
    )

    _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        success_msg=(
            f":raising_hands: Congratulations! You're now using {label} "
            f"{model_phrase} for {family} in voice simulations."
        ),
    )


def _unset_speech_provider(
    *,
    family: str,
    flags: List[str],
    clear_secrets: bool,
    save: Optional[str],
    quiet: bool,
) -> None:
    settings = get_settings()
    with settings.edit(save=save) as edit_ctx:
        for flag in flags:
            setattr(settings, flag, None)
        setattr(settings, f"DEEPEVAL_{family}_MODEL", None)
        if clear_secrets:
            # OPENAI_API_KEY is deliberately left alone: it is shared with the
            # LLM providers, so clearing it here would reach outside speech.
            for flag in flags:
                _, key_field = _SPEECH_PROVIDER_INFO[
                    flag[len("USE_") : -len(f"_{family}")].lower()
                ]
                if key_field != "OPENAI_API_KEY":
                    setattr(settings, key_field, None)

    handled, path, updates = edit_ctx.result

    if _handle_save_result(
        handled=handled,
        path=path,
        updates=updates,
        save=save,
        quiet=quiet,
        updated_msg=(f"Removed {family} environment variables from {{path}}."),
        tip_msg=None,
    ):
        if is_openai_configured():
            print(
                f":raised_hands: OpenAI will still be used for {family} by "
                "default because OPENAI_API_KEY is set."
            )
        else:
            print(
                f"The {family} provider has been unset. Voice simulations "
                "fall back to OpenAI, which needs OPENAI_API_KEY."
            )
