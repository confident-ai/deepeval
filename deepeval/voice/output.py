"""Write simulated conversation audio (per-turn files + combined WAV) to disk."""

import os
import logging
from typing import List, Optional

from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.voice.connectors import audio_utils

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
    combine_audio: bool = True,
) -> None:
    folder = os.path.join(output_dir, run_label)
    if conversation_id is not None:
        folder = os.path.join(folder, conversation_id)
    os.makedirs(folder, exist_ok=True)

    _write_turn_files(test_case.turns, folder)
    if combine_audio:
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
    pcm_parts: List[bytes] = []
    rate: Optional[int] = None
    channels: Optional[int] = None
    for turn in turns:
        if turn.audio is None or _MIME_EXT.get(turn.audio.mimeType) != "wav":
            return None
        try:
            pcm, turn_rate, turn_channels = audio_utils.wav_bytes_to_pcm16(
                turn.audio.get_bytes()
            )
        except ValueError:
            return None
        if rate is None:
            rate, channels = turn_rate, turn_channels
        elif (turn_rate, turn_channels) != (rate, channels):
            return None
        pcm_parts.append(pcm)

    if not pcm_parts:
        return None
    return audio_utils.pcm16_to_wav_bytes(b"".join(pcm_parts), rate, channels)
