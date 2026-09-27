import json

from deepeval.simulator.utils import serialize_turns_for_prompt
from deepeval.test_case import Audio, Turn


def test_serialize_turns_for_prompt_excludes_audio_only():
    turn = Turn(
        role="assistant",
        content="Your replacement is confirmed.",
        audio=Audio.from_bytes(b"voice-audio", "audio/wav"),
        latency_ms=123.0,
        interrupted=False,
        metadata={"channel": "voice"},
    )

    serialized = serialize_turns_for_prompt([turn])
    prompt_turn = json.loads(serialized)[0]

    assert turn.model_dump()["audio"] is not None
    assert "audio" not in turn.model_dump_for_prompt()
    assert "audio" not in prompt_turn
    assert prompt_turn["role"] == "assistant"
    assert prompt_turn["content"] == "Your replacement is confirmed."
    assert prompt_turn["latency_ms"] == 123.0
    assert prompt_turn["interrupted"] is False
    assert prompt_turn["metadata"] == {"channel": "voice"}
