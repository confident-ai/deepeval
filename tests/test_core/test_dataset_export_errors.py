from pathlib import Path

import pytest

from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden


@pytest.mark.parametrize("file_type", ["json", "csv", "jsonl"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("multi_turn", [False, True])
def test_save_as_serialization_error_preserves_destination(
    tmp_path, capsys, file_type, existing, multi_turn
):
    destination = tmp_path / f"dataset.{file_type}"
    original = b"previous export\r\n"
    if existing:
        destination.write_bytes(original)
    valid = {"additional_metadata": {"value": "ok"}}
    invalid = {"additional_metadata": {"value": object()}}
    if multi_turn:
        goldens = [
            ConversationalGolden(scenario="first", **valid),
            ConversationalGolden(scenario="second", **invalid),
        ]
    else:
        goldens = [
            Golden(input="first", **valid),
            Golden(input="second", **invalid),
        ]

    with pytest.raises(TypeError, match="not JSON serializable"):
        EvaluationDataset(goldens).save_as(
            file_type, directory=str(tmp_path), file_name="dataset"
        )

    if existing:
        assert destination.read_bytes() == original
    else:
        assert not destination.exists()
    assert "Evaluation dataset saved" not in capsys.readouterr().out


@pytest.mark.parametrize("file_type", ["json", "csv", "jsonl"])
@pytest.mark.parametrize("existing", [False, True])
def test_save_as_encoding_error_preserves_destination(
    tmp_path, file_type, existing
):
    destination = tmp_path / f"dataset.{file_type}"
    original = b"previous export"
    if existing:
        destination.write_bytes(original)
    dataset = EvaluationDataset([Golden(input="invalid: \ud800")])

    with pytest.raises(UnicodeEncodeError):
        dataset.save_as(file_type, str(tmp_path), file_name="dataset")

    if existing:
        assert destination.read_bytes() == original
    else:
        assert not destination.exists()


@pytest.mark.parametrize("file_type", ["json", "csv", "jsonl"])
@pytest.mark.parametrize("multi_turn", [False, True])
def test_save_as_successfully_replaces_existing_export(
    tmp_path, capsys, file_type, multi_turn
):
    metadata = {"text": '你好,\nquoted "text"', "count": 0, "enabled": False}
    if multi_turn:
        golden = ConversationalGolden(
            scenario="new conversation", additional_metadata=metadata
        )
        field = "scenario"
    else:
        golden = Golden(input="new input", additional_metadata=metadata)
        field = "input"
    destination = tmp_path / f"dataset.{file_type}"
    destination.write_text("previous export", encoding="utf-8")

    result = EvaluationDataset([golden]).save_as(
        file_type, directory=str(tmp_path), file_name="dataset"
    )

    assert Path(result) == destination
    restored = EvaluationDataset()
    getattr(restored, f"add_goldens_from_{file_type}_file")(result)
    assert getattr(restored.goldens[0], field) == getattr(golden, field)
    assert restored.goldens[0].additional_metadata == metadata
    assert "Evaluation dataset saved" in capsys.readouterr().out
