"""Write simulated conversation audio (per-turn files + combined WAV) to disk."""

import os
import logging
from typing import Any, List, Optional

from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.voice.timeline import render_timeline_wav

logger = logging.getLogger(__name__)

# Where recordings land when nobody says otherwise. Namespaced and hidden so a
# simulation run does not scatter audio through the working directory, and kept
# outside the cache folder because clearing a cache should not delete calls the
# user is still listening to.
DEFAULT_VOICE_FOLDER = ".deepeval-voice-simulations"

# Distinguishes "the caller said nothing about where to write", which resolves
# through the env var to a default, from an explicit `None`, which means do not
# write at all.
UNSET: Any = object()


def resolve_output_dir(output_dir: Any = UNSET) -> Optional[str]:
    """Decide where conversation audio goes, or `None` to write none.

    Precedence, highest first: read-only mode, then an explicit `output_dir`
    (including `None` to turn writing off), then `DEEPEVAL_VOICE_FOLDER`, then
    `DEFAULT_VOICE_FOLDER`.
    """
    from deepeval.utils import is_read_only_env

    if is_read_only_env():
        # Honouring an explicit path here would write the one kind of file the
        # user has asked deepeval never to write.
        if output_dir is not UNSET and output_dir is not None:
            logger.warning(
                "READ_ONLY filesystem: not writing voice simulation audio to "
                "%s.",
                output_dir,
            )
        return None

    if output_dir is not UNSET:
        return output_dir

    from deepeval.config.settings import get_settings

    configured = get_settings().DEEPEVAL_VOICE_FOLDER
    return str(configured) if configured is not None else DEFAULT_VOICE_FOLDER


_MIME_EXT = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/opus": "opus",
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/pcm": "pcm",
}


def save_conversation_audio(
    test_case: ConversationalTestCase,
    *,
    output_dir: str,
    run_label: str,
    conversation_id: Optional[str] = None,
    combine_audio_files: bool = True,
) -> None:
    folder = os.path.join(output_dir, run_label)
    if conversation_id is not None:
        folder = os.path.join(folder, conversation_id)
    os.makedirs(folder, exist_ok=True)

    _write_turn_files(test_case.turns, folder)
    if combine_audio_files:
        _write_combined_file(test_case.turns, folder, run_label)


def _write_turn_files(turns: List[Turn], folder: str) -> None:
    turn_number = 0
    for turn in turns:
        if turn.role == "user":
            turn_number += 1
        if turn.audio is None:
            continue
        ext = _MIME_EXT.get(turn.audio.mimeType, "wav")
        filename = f"deepeval-turn-{turn_number}-{turn.role}.{ext}"
        with open(os.path.join(folder, filename), "wb") as f:
            f.write(turn.audio.get_bytes())


def _write_combined_file(
    turns: List[Turn], folder: str, conversation_id: str
) -> None:
    combined = _concat_wav_turns(turns)
    if combined is None:
        logger.warning(
            "Skipping combined audio for %s: turns are not uniform 16-bit "
            "WAV.",
            conversation_id,
        )
        return
    with open(os.path.join(folder, "deepeval-conversation.wav"), "wb") as f:
        f.write(combined)


def _concat_wav_turns(turns: List[Turn]) -> Optional[bytes]:
    has_untimed_audio = any(
        turn.audio is not None and turn.audio.start_time is None
        for turn in turns
    )
    if has_untimed_audio:
        logger.warning(
            "Some turn audio has no start_time; placing untimed clips "
            "sequentially in the combined WAV."
        )
    return render_timeline_wav(turns, require_start_times=False)
