"""Write simulated conversation audio (per-turn files + combined WAV) to disk."""

import os
import logging
from typing import List, Optional

from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.voice.timeline import render_timeline_wav

logger = logging.getLogger(__name__)

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
