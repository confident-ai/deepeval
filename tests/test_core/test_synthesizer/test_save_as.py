import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepeval.dataset.golden import Golden
from deepeval.synthesizer.synthesizer import Synthesizer

_INIT_MODEL_PATHS = [
    "deepeval.synthesizer.synthesizer.initialize_model",
    "deepeval.synthesizer.config.initialize_model",
]


def _make_synthesizer() -> Synthesizer:
    fake_model = MagicMock()
    fake_model.get_model_name.return_value = "fake-model"
    fake_model.generate.return_value = ("fake", 0.0)
    fake_model.a_generate = AsyncMock(return_value=("fake", 0.0))

    with patch(_INIT_MODEL_PATHS[0], return_value=(fake_model, True)), patch(
        _INIT_MODEL_PATHS[1], return_value=(fake_model, True)
    ):
        synthesizer = Synthesizer(model=fake_model, async_mode=False)

    synthesizer.synthetic_goldens = [
        Golden(input="What is DeepEval?", expected_output="An eval library.")
    ]
    return synthesizer


@pytest.mark.parametrize(
    "file_type, expected_extension",
    [("JSON", "json"), ("Csv", "csv"), ("JSONL", "jsonl")],
)
def test_save_as_accepts_file_type_in_any_case(
    tmp_path, file_type, expected_extension
):
    synthesizer = _make_synthesizer()

    path = synthesizer.save_as(
        file_type=file_type,
        directory=str(tmp_path),
        file_name="goldens",
        quiet=True,
    )

    assert os.path.basename(path) == f"goldens.{expected_extension}"
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_save_as_invalid_file_type_lists_the_valid_ones(tmp_path):
    synthesizer = _make_synthesizer()

    with pytest.raises(ValueError) as exc_info:
        synthesizer.save_as(file_type="xml", directory=str(tmp_path))

    assert (
        str(exc_info.value)
        == "Invalid file type. Available file types to save as: "
        "csv, json, jsonl"
    )
