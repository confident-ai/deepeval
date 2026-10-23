import csv
import json

import pytest

from deepeval.dataset import ConversationalGolden, EvaluationDataset
from deepeval.metrics.transcription_accuracy.transcription_accuracy import (
    get_transcribed_exchanges,
)
from deepeval.test_case import Audio, ConversationalTestCase, Turn


@pytest.mark.parametrize("file_type", ["json", "csv", "jsonl"])
@pytest.mark.parametrize("include_test_cases", [False, True])
def test_save_as_preserves_voice_turns(tmp_path, file_type, include_test_cases):
    turns = [
        Turn(
            role="user",
            content="Book a flight to Tokyo.",
            audio=Audio.from_bytes(
                b"caller audio",
                "audio/wav",
                sampleRate=24000,
                encoding="wav",
                duration=1.5,
                start_time=0.0,
            ),
        ),
        Turn(
            role="assistant",
            content="When would you like to travel?",
            audio=Audio(url="https://example.com/reply.wav"),
            latency_ms=250.5,
            interrupted=True,
            provider_transcript="Book a flight to Kyoto.",
        ),
        Turn(role="user", content="Tomorrow."),
        Turn(
            role="assistant",
            content="",
            latency_ms=0.0,
            interrupted=False,
            provider_transcript="",
        ),
    ]
    scenario = "Book a flight"
    if include_test_cases:
        dataset = EvaluationDataset()
        dataset.add_test_case(
            ConversationalTestCase(scenario=scenario, turns=turns)
        )
    else:
        dataset = EvaluationDataset(
            goldens=[ConversationalGolden(scenario=scenario, turns=turns)]
        )

    path = dataset.save_as(
        file_type,
        directory=str(tmp_path),
        include_test_cases=include_test_cases,
    )
    with open(path, encoding="utf-8", newline="") as exported_file:
        if file_type == "csv":
            exported_turns = json.loads(
                next(csv.DictReader(exported_file))["turns"]
            )
        elif file_type == "jsonl":
            exported_turns = json.loads(exported_file.readline())["turns"]
        else:
            exported_turns = json.load(exported_file)[0]["turns"]

    assert not {"_local", "_filename", "_id"}.intersection(
        exported_turns[0]["audio"]
    )
    assert not {
        "audio",
        "latency_ms",
        "interrupted",
        "provider_transcript",
    }.intersection(exported_turns[2])

    loaded = EvaluationDataset()
    getattr(loaded, f"add_goldens_from_{file_type}_file")(path)
    loaded_turns = loaded.goldens[0].turns

    assert len(loaded_turns) == len(turns)
    for original, restored in zip(turns, loaded_turns):
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.latency_ms == original.latency_ms
        assert restored.interrupted == original.interrupted
        assert restored.provider_transcript == original.provider_transcript
        if original.audio is None:
            assert restored.audio is None
        else:
            assert restored.audio is not None
            for field in (
                "dataBase64",
                "mimeType",
                "url",
                "sampleRate",
                "encoding",
                "duration",
                "start_time",
            ):
                assert getattr(restored.audio, field) == getattr(
                    original.audio, field
                )
            if original.audio.dataBase64 is not None:
                assert restored.audio.get_bytes() == original.audio.get_bytes()

    assert get_transcribed_exchanges(loaded_turns) == get_transcribed_exchanges(
        turns
    )
