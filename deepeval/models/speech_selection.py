"""Provider selection for speech models.

TTS and STT mirror the LLM mechanism in `deepeval.metrics.utils`: a `USE_*`
flag picks the provider, and the model name comes from the family-level
`DEEPEVAL_TTS_MODEL` / `DEEPEVAL_STT_MODEL` unless one is passed in code. With
no flag set, both families fall back to OpenAI.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

from deepeval.config.settings import get_settings
from deepeval.key_handler import KEY_FILE_HANDLER, SpeechKeyValues
from deepeval.models.base_model import DeepEvalBaseSTT, DeepEvalBaseTTS

# Flag -> class name, in the order they are checked. Named rather than
# imported because `deepeval.models.tts` reaches back into `deepeval.voice`.
_TTS_PROVIDERS: Tuple[Tuple[str, str], ...] = (
    ("USE_OPENAI_TTS", "OpenAITTSModel"),
    ("USE_ELEVENLABS_TTS", "ElevenLabsTTSModel"),
    ("USE_CARTESIA_TTS", "CartesiaTTSModel"),
    ("USE_DEEPGRAM_TTS", "DeepgramTTSModel"),
)
_STT_PROVIDERS: Tuple[Tuple[str, str], ...] = (
    ("USE_OPENAI_STT", "OpenAISTTModel"),
    ("USE_ELEVENLABS_STT", "ElevenLabsSTTModel"),
    ("USE_CARTESIA_STT", "CartesiaSTTModel"),
    ("USE_DEEPGRAM_STT", "DeepgramSTTModel"),
    ("USE_ASSEMBLYAI_STT", "AssemblyAISTTModel"),
)


def _from_keystore(field: str) -> Optional[str]:
    member = SpeechKeyValues.__members__.get(field)
    if member is None:
        return None
    return KEY_FILE_HANDLER.fetch_data(member)


def _flag_enabled(field: str) -> bool:
    if getattr(get_settings(), field, None):
        return True
    value = _from_keystore(field)
    return value.lower() == "yes" if value is not None else False


def _model_name(field: str, override: Optional[str]) -> Optional[str]:
    if override is not None:
        return override
    return getattr(get_settings(), field, None) or _from_keystore(field)


def _selected(providers: Tuple[Tuple[str, str], ...]) -> str:
    for flag, class_name in providers:
        if _flag_enabled(flag):
            return class_name
    # No flag set: OpenAI, matching `initialize_model()`'s LLM fallback.
    return providers[0][1]


def describe_tts_selection() -> Tuple[str, Optional[str]]:
    """The TTS class name and model name selected by the environment.

    Reported without constructing anything, so it stays answerable on a
    machine with no speech API key set.
    """
    return _selected(_TTS_PROVIDERS), _model_name("DEEPEVAL_TTS_MODEL", None)


def describe_stt_selection() -> Tuple[str, Optional[str]]:
    """The STT class name and model name selected by the environment."""
    return _selected(_STT_PROVIDERS), _model_name("DEEPEVAL_STT_MODEL", None)


def initialize_tts_model(
    model: Optional[Union[str, DeepEvalBaseTTS]] = None,
) -> DeepEvalBaseTTS:
    """The TTS model for this run.

    A model object is used as-is, a string is the model name for the
    flag-selected provider, and `None` falls back to `DEEPEVAL_TTS_MODEL`.
    """
    if model is not None and not isinstance(model, str):
        return model

    from deepeval.models import tts

    class_name = _selected(_TTS_PROVIDERS)
    return getattr(tts, class_name)(
        model=_model_name("DEEPEVAL_TTS_MODEL", model)
    )


def initialize_stt_model(
    model: Optional[Union[str, DeepEvalBaseSTT]] = None,
) -> DeepEvalBaseSTT:
    """The STT model for this run.

    A model object is used as-is, a string is the model name for the
    flag-selected provider, and `None` falls back to `DEEPEVAL_STT_MODEL`.
    """
    if model is not None and not isinstance(model, str):
        return model

    from deepeval.models import stt

    class_name = _selected(_STT_PROVIDERS)
    return getattr(stt, class_name)(
        model=_model_name("DEEPEVAL_STT_MODEL", model)
    )
