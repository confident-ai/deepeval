from __future__ import annotations

import pytest

from deepeval.cli.test.command import check_if_valid_file


def test_accepts_file_without_test_prefix(tmp_path):
    eval_file = tmp_path / "judgment_eval.py"
    eval_file.write_text("def test_something():\n    assert True\n")

    check_if_valid_file(str(eval_file))


def test_rejects_nonexistent_path(tmp_path):
    missing = tmp_path / "does_not_exist.py"

    with pytest.raises(
        ValueError, match="neither a valid file nor a directory"
    ):
        check_if_valid_file(str(missing))
